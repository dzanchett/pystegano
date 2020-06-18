from distutils.core import setup
setup(
  name = 'pystegano',
  packages = ['pystegano'],
  version = '0.1',
  license='MIT',
  description = 'Steganography tools for python',
  author = 'Diego Zanchett',
  author_email = 'diego.zanchett@aluno.cefet-rj.br',
  url = 'https://github.com/dzanchett/pystegano',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['steganography'],
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
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