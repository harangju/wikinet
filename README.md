# WikiNet
This repository contains code for analysis used in [Ju et al. (2020)](https://arxiv.org/abs/2010.08381).

## Getting started
1. In the terminal, `git clone https://github.com/harangju/wikinet.git`
2. `cd wikinet`
3. `conda env create -f environment.yml`
    * Download [anaconda](https://www.anaconda.com).
4. `jupyter notebook`

## Data
Wikipedia XML dumps are available at https://dumps.wikimedia.org/enwiki. Only two files are required for reproduction: (1) enwiki-DATE-pages-articles-multistream.xml.bz2 and (2) enwiki-DATE-pages-articles-multistream-index.txt.bz2, where DATE is the date of the dump. Both files are multistreamed versions of the zipped files, which allow the user to access an article without unpacking the whole file. In this study, we used the archived zipped file from August 1, 2019, which is available [here](https://www.dropbox.com/sh/kwsubhwf787p74k/AAA0Wf_3-SZggcvRYdrdzXBba?dl=0).
