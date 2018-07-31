#!/usr/bin/env bash

if [ ! -e datasets4fastText ]; then
    git clone https://github.com/mwydmuch/datasets4fastText.git
    cd datasets4fastText
    git checkout with_features_values
    cd ..
fi

bash datasets4fastText/xml_repo/get_wikiLSHTC.sh
bash datasets4fastText/xml_repo/get_amazon.sh
bash datasets4fastText/xml_repo/get_wiki10.sh
bash datasets4fastText/xml_repo/get_eurlex.sh