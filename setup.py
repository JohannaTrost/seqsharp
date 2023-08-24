from setuptools import setup, find_packages

setup(
    name='seqsharp',
    version='0.1.0',
    description='Sequence evolution Simulations Have A Real(ism) Problem - '
                'an accurate discriminator of simulated and empirical MSAs '
                'using CNNs',
    author='Johanna Trost',
    author_email='johanna.trost.1997@gmail.com',
    license='CeCIL',
    python_requires='>=3.6, <=3.10',
    packages=find_packages(include=['seqsharp', 'seqsharp.*']),
    # include package data such as the pretrained_models (all non .py files)
    package_dir={'seqsharp': 'seqsharp'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'seqsharp = seqsharp.__main__:main'
        ]
    }
)
