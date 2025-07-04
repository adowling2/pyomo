#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess

from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import

request = attempt_import('urllib.request')[0]
urllib_error = attempt_import('urllib.error')[0]
ssl = attempt_import('ssl')[0]
zipfile = attempt_import('zipfile')[0]
tarfile = attempt_import('tarfile')[0]
gzip = attempt_import('gzip')[0]
distro, distro_available = attempt_import('distro')

logger = logging.getLogger('pyomo.common.download')

DownloadFactory = pyomo.common.Factory('library downloaders')


class FileDownloader(object):
    _os_version = None

    def __init__(self, insecure=False, cacert=None):
        self._fname = None
        self.target = None
        self.insecure = insecure
        self.cacert = cacert
        if cacert is not None:
            if not self.cacert or not os.path.isfile(self.cacert):
                raise RuntimeError(
                    "cacert='%s' does not refer to a valid file." % (self.cacert,)
                )

    @classmethod
    def get_sysinfo(cls):
        """Return a tuple (platform_name, bits) for the current system

        Returns
        -------
           platform_name (str): lower case, usually in {linux, windows,
              cygwin, darwin}.
           bits (int): OS address width in {32, 64}
        """

        system = platform.system().lower()
        for c in '.-_':
            system = system.split(c)[0]
        bits = 64 if sys.maxsize > 2**32 else 32
        return system, bits

    @classmethod
    def _get_distver_from_os_release(cls):
        dist = ''
        ver = ''
        with open('/etc/os-release', 'rt') as FILE:
            for line in FILE:
                line = line.strip()
                if not line:
                    continue
                key, val = line.lower().split('=')
                if val[0] == val[-1] and val[0] in '"\'':
                    val = val[1:-1]
                if key == 'id':
                    dist = val
                elif key == 'version_id':
                    ver = val
        return cls._map_linux_dist(dist), ver

    @classmethod
    def _get_distver_from_redhat_release(cls):
        # RHEL6 did not include /etc/os-release
        with open('/etc/redhat-release', 'rt') as FILE:
            dist = FILE.readline().lower().strip()
            ver = ''
            for word in dist.split():
                if re.match(r'^[0-9\.]+', word):
                    ver = word
                    break
        return cls._map_linux_dist(dist), ver

    @classmethod
    def _get_distver_from_lsb_release(cls):
        lsb_release = shutil.which('lsb_release')
        dist = subprocess.run(
            [lsb_release, '-si'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        ver = subprocess.run(
            [lsb_release, '-sr'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return cls._map_linux_dist(dist.stdout), ver.stdout.strip()

    @classmethod
    def _get_distver_from_distro(cls):
        return distro.id(), distro.version(best=True)

    @classmethod
    def _map_linux_dist(cls, dist):
        # Some distributions end up reporting longer strings or strings
        # with spaces (e.g., RHEL6 reports 'red hat enterprise')
        dist = dist.lower().strip().replace(' ', '')
        _map = [
            ('redhat', 'rhel'),
            'fedora',
            'ubuntu',  # implicitly maps kubuntu / xubuntu
            'debian',
            # Additional RHEL (Fedora) spins
            'centos',
            'rocky',
            'almalinux',
            'scientific',
            # Additional Ubuntu (Debian) spins
            'linuxmint',
        ]
        for key in _map:
            if key.__class__ is tuple:
                key, val = key
            else:
                val = key
            if key in dist:
                return val
        return dist

    @classmethod
    def _get_os_version(cls):
        _os = cls.get_sysinfo()[0]
        if _os == 'linux':
            if distro_available:
                dist, ver = cls._get_distver_from_distro()
            elif os.path.exists('/etc/redhat-release'):
                dist, ver = cls._get_distver_from_redhat_release()
            elif shutil.which('lsb_release'):
                dist, ver = cls._get_distver_from_lsb_release()
            elif os.path.exists('/etc/os-release'):
                # Note that (at least on centos), os_release is an
                # imprecise version string
                dist, ver = cls._get_distver_from_os_release()
            else:
                dist, ver = '', ''
            return dist, ver
        elif _os == 'darwin':
            return 'macos', platform.mac_ver()[0]
        elif _os == 'windows':
            return 'win', platform.win32_ver()[0]
        else:
            return '', ''

    @classmethod
    def get_os_version(cls, normalize=True):
        """Return a standardized representation of the OS version

        This method was designed to help identify compatible binaries,
        and will return strings similar to:

          - rhel6
          - fedora24
          - ubuntu18.04
          - macos10.13
          - win10

        Parameters
        ----------
        normalize : bool, optional
            If True (the default) returns a simplified normalized string
            (e.g., `'rhel7'`) instead of the raw (os, version) tuple
            (e.g., `('centos', '7.7.1908')`)

        """
        if FileDownloader._os_version is None:
            FileDownloader._os_version = cls._get_os_version()

        if not normalize:
            return FileDownloader._os_version

        _os, _ver = FileDownloader._os_version
        _map = {
            'almalinux': 'rhel',
            'centos': 'rhel',
            'rocky': 'rhel',
            'scientific': 'rhel',
            'kubuntu': 'ubuntu',
            'linuxmint': 'ubuntu',
            'xubuntu': 'ubuntu',
        }
        if _os in _map:
            _os = _map[_os]

        if _os in {'ubuntu', 'macos', 'win'}:
            return _os + ''.join(_ver.split('.')[:2])
        else:
            return _os + _ver.split('.')[0]

    @deprecated("get_url() is deprecated. Use get_platform_url()", version='5.6.9')
    def get_url(self, urlmap):
        return self.get_platform_url(urlmap)

    def get_platform_url(self, urlmap):
        """Select the url for this platform

        Given a `urlmap` dict that maps the platform name (from
        `FileDownloader.get_sysinfo()`) to a platform-specific URL,
        return the URL that matches the current platform.

        Parameters
        ----------
        urlmap: dict
            Map of platform name (e.g., `linux`, `windows`, `cygwin`,
            `darwin`) to URL

        """
        system, bits = self.get_sysinfo()
        url = urlmap.get(system, None)
        if url is None:
            raise RuntimeError(
                "cannot infer the correct url for platform '%s'" % (platform,)
            )
        return url

    def create_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            '--insecure',
            action='store_true',
            dest='insecure',
            default=False,
            help="Disable all SSL verification",
        )
        parser.add_argument(
            '--cacert',
            action='store',
            dest='cacert',
            default=None,
            help="Use CACERT as the file of certificate authorities "
            "to verify peers.",
        )
        parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            dest='verbose',
            default=False,
            help="Verbose output when download fails",
        )
        return parser

    def parse_args(self, argv):
        parser = self.create_parser()
        parser.add_argument(
            'target',
            nargs="?",
            default=None,
            help="Target destination directory or filename",
        )
        parser.parse_args(argv, self)
        if self.cacert is not None:
            if not self.cacert or not os.path.isfile(self.cacert):
                raise RuntimeError(
                    "--cacert='%s' does not refer to a valid file." % (self.cacert,)
                )

    def set_destination_filename(self, default):
        if self.target is not None:
            self._fname = self.target
        else:
            self._fname = envvar.PYOMO_CONFIG_DIR
            if not os.path.isdir(self._fname):
                os.makedirs(self._fname)
        if os.path.isdir(self._fname):
            self._fname = os.path.join(self._fname, default)
        targetDir = os.path.dirname(self._fname)
        if not os.path.isdir(targetDir):
            os.makedirs(targetDir)

    def destination(self):
        return self._fname

    def retrieve_url(self, url):
        """Return the contents of a URL as an io.BytesIO object"""
        ctx = ssl.create_default_context()
        if self.cacert:
            ctx.load_verify_locations(cafile=self.cacert)
        if self.insecure:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        try:
            fetch = request.urlopen(url, context=ctx)
        except urllib_error.HTTPError as e:
            if e.code != 403:  # Forbidden
                raise
            fetch = None
        if fetch is None:
            # This is a fix implemented if we get stuck behind server
            # security features (attempting to block "bot" agents).
            # We are setting a known user-agent to get around that.
            req = request.Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
            fetch = request.urlopen(req, context=ctx)
        ans = fetch.read()
        logger.info("  ...downloaded %s bytes" % (len(ans),))
        return ans

    def get_file(self, url, binary):
        if self._fname is None:
            raise DeveloperError(
                "target file name has not been initialized "
                "with set_destination_filename"
            )
        with open(self._fname, 'wb' if binary else 'wt') as FILE:
            raw_file = self.retrieve_url(url)
            if binary:
                FILE.write(raw_file)
            else:
                FILE.write(raw_file.decode())
            logger.info("  ...wrote %s bytes" % (len(raw_file),))

    def get_binary_file(self, url):
        """Retrieve the specified url and write as a binary file"""
        return self.get_file(url, binary=True)

    def get_text_file(self, url):
        """Retrieve the specified url and write as a text file"""
        return self.get_file(url, binary=False)

    def get_binary_file_from_zip_archive(self, url, srcname):
        if self._fname is None:
            raise DeveloperError(
                "target file name has not been initialized "
                "with set_destination_filename"
            )
        with open(self._fname, 'wb') as FILE:
            zipped_file = io.BytesIO(self.retrieve_url(url))
            raw_file = zipfile.ZipFile(zipped_file).open(srcname).read()
            FILE.write(raw_file)
            logger.info("  ...wrote %s bytes" % (len(raw_file),))

    def get_zip_archive(self, url, dirOffset=0):
        if self._fname is None:
            raise DeveloperError(
                "target file name has not been initialized "
                "with set_destination_filename"
            )
        if os.path.exists(self._fname) and not os.path.isdir(self._fname):
            raise RuntimeError(
                "Target directory (%s) exists, but is not a directory" % (self._fname,)
            )
        zip_file = zipfile.ZipFile(io.BytesIO(self.retrieve_url(url)))
        # Simple sanity checks
        for info in zip_file.infolist():
            f = info.filename
            if f[0] in '\\/' or '..' in f or os.path.isabs(f):
                logger.error(
                    "malformed (potentially insecure) filename (%s) "
                    "found in zip archive.  Skipping file." % (f,)
                )
                continue
            target = self._splitpath(f)
            if len(target) <= dirOffset:
                if f[-1] != '/':
                    logger.warning(
                        "Skipping file (%s) in zip archive due to dirOffset" % (f,)
                    )
                continue
            info.filename = target[-1] + '/' if f[-1] == '/' else target[-1]
            zip_file.extract(f, os.path.join(self._fname, *tuple(target[dirOffset:-1])))

    def get_tar_archive(self, url, dirOffset=0):
        if self._fname is None:
            raise DeveloperError(
                "target file name has not been initialized "
                "with set_destination_filename"
            )
        if os.path.exists(self._fname) and not os.path.isdir(self._fname):
            raise RuntimeError(
                "Target directory (%s) exists, but is not a directory" % (self._fname,)
            )

        def filter_fcn(info):
            # this mocks up the `tarfile` filter introduced in Python
            # 3.12 and backported to later releases of Python (e.g.,
            # 3.8.17, 3.9.17, 3.10.12, and 3.11.4)
            f = info.name
            if os.path.isabs(f) or '..' in f or f.startswith(('/', os.sep)):
                logger.error(
                    "malformed or potentially insecure filename (%s).  "
                    "Skipping file." % (f,)
                )
                return False
            target = self._splitpath(f)
            if len(target) <= dirOffset:
                if not info.isdir():
                    logger.warning(
                        "Skipping file (%s) in tar archive due to dirOffset." % (f,)
                    )
                return False
            info.name = f = '/'.join(target[dirOffset:])
            target = os.path.realpath(os.path.join(dest, f))
            try:
                if os.path.commonpath([target, dest]) != dest:
                    logger.error(
                        "potentially insecure filename (%s) resolves outside target "
                        "directory.  Skipping file." % (f,)
                    )
                    return False
            except ValueError:
                # commonpath() will raise ValueError for paths that
                # don't have anything in common (notably, when files are
                # on different drives on Windows)
                logger.error(
                    "potentially insecure filename (%s) resolves outside target "
                    "directory.  Skipping file." % (f,)
                )
                return False
            # Strip high bits & group/other write bits
            info.mode &= 0o755
            return True

        if sys.version_info[:2] < (3, 12):
            tar_args = {}
        else:
            # Add a tar filter to suppress deprecation warning in Python 3.12+
            #
            # Note that starting in Python 3.14 the default is 'data',
            # however, this method is already filtering for many of the
            # things that 'data' would catch (and raise an exception
            # for).  For backwards compatibility, we will use the more
            # permissive "tar" filter.
            tar_args = {'filter': tarfile.tar_filter}
        with tarfile.open(fileobj=io.BytesIO(self.retrieve_url(url))) as TAR:
            dest = os.path.realpath(self._fname)
            TAR.extractall(dest, filter(filter_fcn, TAR.getmembers()), **tar_args)

    def get_gzipped_binary_file(self, url):
        if self._fname is None:
            raise DeveloperError(
                "target file name has not been initialized "
                "with set_destination_filename"
            )
        with open(self._fname, 'wb') as FILE:
            gzipped_file = io.BytesIO(self.retrieve_url(url))
            raw_file = gzip.GzipFile(fileobj=gzipped_file).read()
            FILE.write(raw_file)
            logger.info("  ...wrote %s bytes" % (len(raw_file),))

    def _splitpath(self, path):
        components = []
        head, tail = os.path.split(os.path.normpath(path))
        while head != path:
            if tail:
                components.append(tail)
            path = head
            head, tail = os.path.split(path)
        if head:
            components.append(head)
        components.reverse()
        return components
