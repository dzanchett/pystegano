from setuptools import setup

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name = 'pystegano',
  packages = ['pystegano'],
  version = '0.3.1',
  license='MIT',
  description = 'Steganography tools for Python',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Diego Zanchett',
  author_email = 'diego.zanchett@aluno.cefet-rj.br',
  url = 'https://github.com/dzanchett/pystegano',
  download_url = 'https://github.com/dzanchett/pystegano/archive/v_0.3.1.tar.gz',
  keywords = ['steganography'],
  install_requires=[
          'scipy',
          'numpy',
          'pandas',
          'scikit-image',
          'bitarray',
          'opencv-python',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)