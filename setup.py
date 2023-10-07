#!/usr/bin/env python

from pip._internal.req.req_file import parse_requirements
from setuptools import setup, find_packages, find_namespace_packages
from os.path import join
import sys
import re
from pip._internal.req import parse_requirements

sys.path.append('.')
# from ssdwsn.data.net import VERSION
install_reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.requirement) for ir in install_reqs]
# scripts = [ join( 'bin', filename ) for filename in [ 'ssdwsn' ] ]
distname = 'ssdwsn'
setup(
        name=distname,
        version=1.0,
        description='ssdwsn simulation',
        author='Saeedi',
        author_email='',    
        packages=find_namespace_packages(),
        package_dir={"": '.'},
        # packages= ['ssdwsn', 'ssdwsn.app', 'ssdwsn.ctrl', 'ssdwsn.data', 'ssdwsn.openflow', 'ssdwsn.util'],
        package_data={},
        keywords='',    
        license='',
        include_package_data=True,
        install_requires=reqs,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Healthcare Industry',
            'Topic :: Text Processing',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: MacOS',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.0',
            'Programming Language :: Python :: 3.1',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7'
        ],
        # scripts=scripts,
    )
