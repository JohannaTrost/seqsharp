from setuptools import setup

setup(
    name='seqsharp',
    version='0.1.0',
    packages=['seqsharp'],
    entry_points={
        'console_scripts': [
            'seqsharp = seqsharp.__main__:main'
        ]
    })
