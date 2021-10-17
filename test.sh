#!/bin/sh

scp -i ~/Documents/Misc/Keys/deeplearning.pem ubuntu@deeplearning:/home/ubuntu/PyCharmProjects/DasGymnasium/save/policy /Users/maincharacter/Dropbox/PyCharmProjects/DasGymnasium/download
python berndtzl-player.py ./download/policy 16
