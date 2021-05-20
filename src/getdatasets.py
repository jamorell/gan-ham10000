import urllib.request
import zipfile
import shutil
import os

def get_dataset_from_url(dataset_name, dataset_url):
  try:
    #Getting filename
    firstpos = dataset_url.rfind("/")
    lastpos = len(dataset_url)
    filename = dataset_url[firstpos+1 : lastpos]
    #
    os.makedirs(dataset_name)
    contents = urllib.request.urlopen(dataset_url).read()
    f = open("./" + dataset_name + "/" + filename, "wb")
    f.write(contents)
    f.close()

    with zipfile.ZipFile("./" + dataset_name + "/" + filename, 'r') as zip_ref:
      zip_ref.extractall("./" + dataset_name + "/")

  except Exception as e:
    print(e)


def remove_dataset_folder(dataset_folder):
  shutil.rmtree(dataset_folder)
  print("Done!")
