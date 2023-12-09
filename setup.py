from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='ssdwsn',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/moalsaeedi/ssdwsn',
    author='Mohd Alsaeedi',
    description='',
    install_requires=requirements,
    entry_points=dict(console_scripts=[
        'ssdwsn=app:main'
    ])
)
