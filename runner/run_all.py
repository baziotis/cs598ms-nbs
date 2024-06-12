import subprocess
from glob import glob
import run_nb
import argparse
import os
import sys
import pathlib

parser = argparse.ArgumentParser(description='Run all benchmarks')
parser.add_argument('--alt', choices=['modin', 'koalas', 'analytical'], help='Pandas alternative. If left unspecified, it uses regular pandas. Otherwise, it can either modin, or koalas, or the analytical model which is a hybrid of pandas and modin.')
parser.add_argument('--cores', type=int, metavar='NUM_CORES', help='Number of cores to use with modin or koalas. Valid (and required) only if --alt has been specified.')
parser.add_argument('--less_replication', action='store_true', help='Less replication of data.')
parser.add_argument('--measure_mem', action='store_true', help='Measure memory consumption (only works for pandas and modin, not koalas).')
parser.add_argument('--dataset_size', type=int, metavar='DATA_MB', help='Replicate or truncate datasets to a total of the specified size (in MB).')
parser.add_argument('--measure_data', action='store_true', help='Measure the memory usage and conversion cost.')

args = parser.parse_args()

# Some (hand-wavy) validation
msg=None

if args.cores is not None:
  if args.cores < 2:
    msg = "--cores option must be at least 2"
  if args.alt is None:
    msg = "--cores can be specified only if --alt has been specified"

if args.alt is not None:
  if args.alt == "koalas" and args.measure_mem:
    msg = "--measure_mem can be specified only with pandas and modin"
  if args.cores is None:
    msg = "When specifying a pandas alternative, you need to specify --cores"

if msg is not None:
  print("ERROR:", msg)
  parser.print_help()
  sys.exit(1)

# Put a version file into the "stats" folder
assert os.path.isdir("./stats")
ver_file = open('stats/.version', 'w+')
VER_pandas = "pandas" if args.alt is None else args.alt
VER_repl = "repl_LESS" if args.less_replication else "repl_STD"
VER_sliced_exec = "mem_ON" if args.measure_mem else "mem_OFF"
VER_dataset_size = str(args.dataset_size) + " MB" if args.dataset_size else "original_size"
VER="-".join((VER_pandas, VER_repl, VER_sliced_exec, VER_dataset_size))
ver_file.write(VER)
ver_file.close()

prefix = str(pathlib.Path('../notebooks').resolve())

nbs = [
  #!M1G#"aieducation/what-course-are-you-going-to-take",
  "akshaypetkar/supermarket-sales-analysis",
  #!M256#"ampiiere/animal-crossing-villager-popularity-analysis",
  #!M256#"arimishabirin/globalsalary-simple-eda",
  #!M#"artgor/eda-and-models",
  "beratozmen/clash-of-clans-exploratory-data-analysis",
  #!M256#"brianmendieta/data-cleaning-plus-eda",#!A#
  "carlmcbrideellis/simple-eda-of-kaggle-grandmasters-scheduled",
  #!M#"corazzon/how-to-use-pandas-filter-in-survey-eda",
  #!P256#"dataranch/supermarket-sales-prediction-xgboost-fastai",
  #!M#"deffro/eda-is-fun",
  #!M#"erikbruin/nlp-on-student-writing-eda",
  "gksriharsha/eda-speedtests",
  #!P256#"ibtesama/getting-started-with-a-movie-recommendation-system",#!A#
  "itzsanju/eda-airline-dataset",
  #!M#"jagangupta/stop-the-s-toxic-comments-eda",
  "josecode1/billionaires-statistics-2023",
  #!M#"joshuaswords/netflix-data-visualization",
  "jyotsananegi/melbourne-housing-snapshot-eda",
  #!M#"kabure/extensive-usa-youtube-eda",
  #!P256#"kanncaa1/dataiteam-titanic-eda",
  "kenjee/titanic-project-example",
  #!P256#"khoongweihao/covid-19-novel-coronavirus-eda-forecasting-cases",
  #!M1G#"kimtaehun/simple-preprocessing-for-time-series-prediction",
  "kkhandekar/environmental-vs-ai-startups-india-eda",
  #!M#"korfanakis/housing-in-london-eda-with-pandas-and-gif",
  #!M256#"lextoumbourou/feedback3-eda-hf-custom-trainer-sift",
  #!P256#"macespinoza/simple-eda-with-python-pandas-data-avocado-paltas",#!A#
  #!P256#"madhurpant/beautiful-kaggle-2022-analysis",
  "madseth/customer-shopping-trends-dataset-eda",
  "mathewvondersaar/analysis-of-student-performance",
  "mikedelong/python-eda-with-kdes",
  #!M1G#"mpwolke/just-you-wait-rishi-sunak",
  "muhammadawaistayyab/used-cars-in-pakistan-stats",
  "natigmamishov/eda-with-pandas-on-telecom-churn-dataset",
  "nickwan/creating-player-stats-using-tracking-data",
  #!M1G#"nicoleashley/iit-admission-eda",#!A#
  "paultimothymooney/kaggle-survey-2022-all-results",
  "pmarcelino/comprehensive-data-exploration-with-python",
  "qnqfbqfqo/electric-vehicle-landscape-in-washington-state",
  "robikscube/big-data-bowl-comprehensive-eda-with-pandas",
  #!M1G#"roopacalistus/exploratory-data-analysis-retail-supermarket",
  "roopacalistus/retail-supermarket-store-analysis",
  "roopahegde/cryptocurrency-price-correlation",
  #!M256#"saisandeepjallepalli/adidas-retail-eda-data-visualization",
  "sandhyakrishnan02/indian-startup-growth-analysis",
  #!M#"saniaks/melbourne-house-price-eda",
  #!M#"sanket7994/imdb-dataset-eda-project",
  #!M#"shivavashishtha/zomato-eda-tutorial",
  #!M256#"spscientist/student-performance-in-exams",
  #!M#"sunnybiswas/eda-on-airline-dataset",
  #!M#"tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model",
  #!M#"vanguarde/h-m-eda-first-look",
  "vatsalmavani/music-recommendation-system-using-spotify-dataset",
  #!M#"vbmokin/automatic-eda-with-pandas-profiling-2-9-09-2020",
  #!M1G#"viviktpharale/house-price-prediction-eda-linear-ridge-lasso",
  #!M#"willkoehrsen/start-here-a-gentle-introduction",
  #!M1G#"xokent/cyber-security-attack-eda",
  "yuliagm/talkingdata-eda-plus-time-patterns"
]

for nb in nbs:
  kernel_user = nb.split('/')[0]
  kernel_slug = nb.split('/')[1]
  full_path = prefix+"/"+nb
  print(f"--- RUNNING: {kernel_user}/{kernel_slug}")
  succ = run_nb.run_nb_paper(full_path, args)
  assert succ
  res = subprocess.run(["mv", "stats.json", f"stats/{kernel_user}_{kernel_slug}.json"])
  assert res.returncode == 0
