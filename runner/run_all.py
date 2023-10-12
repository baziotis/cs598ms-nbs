import subprocess
from glob import glob
import run_nb
import argparse
import os
import sys
import pathlib

parser = argparse.ArgumentParser(description='Run all benchmarks')
parser.add_argument('--alt', choices=['modin', 'koalas'], help='Pandas alternative. If left unspecified, it uses regular pandas. Otherwise, it can either modin or koalas')
parser.add_argument('--cores', type=int, metavar='NUM_CORES', help='Number of cores to use with modin or koalas. Valid (and required) only if --alt has been specified.')
parser.add_argument('--less_replication', action='store_true', help='Less replication of data.')
parser.add_argument('--measure_mem', action='store_true', help='Measure memory consumption (only works for pandas and modin, not koalas).')

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
VER="-".join((VER_pandas, VER_repl, VER_sliced_exec))
ver_file.write(VER)
ver_file.close()

prefix = str(pathlib.Path('../notebooks').resolve())

nbs_we_hit = [
  "lextoumbourou/feedback3-eda-hf-custom-trainer-sift",
  "paultimothymooney/kaggle-survey-2022-all-results",
  "dataranch/supermarket-sales-prediction-xgboost-fastai",
  "kkhandekar/environmental-vs-ai-startups-india-eda",
  "ampiiere/animal-crossing-villager-popularity-analysis",
  "aieducation/what-course-are-you-going-to-take",
  "saisandeepjallepalli/adidas-retail-eda-data-visualization",
  "joshuaswords/netflix-data-visualization",
  "spscientist/student-performance-in-exams",
 "ibtesama/getting-started-with-a-movie-recommendation-system",
]

nbs_we_dont = [
  "nickwan/creating-player-stats-using-tracking-data",
  "erikbruin/nlp-on-student-writing-eda",
  "madhurpant/beautiful-kaggle-2022-analysis",
  "pmarcelino/comprehensive-data-exploration-with-python",
  "gksriharsha/eda-speedtests",
  "mpwolke/just-you-wait-rishi-sunak",
  "sanket7994/imdb-dataset-eda-project",
  "roopacalistus/retail-supermarket-store-analysis",
  "sandhyakrishnan02/indian-startup-growth-analysis",
  "roopacalistus/exploratory-data-analysis-retail-supermarket"
]

# TODO: Merge these into one list. There's no rewriter no so the concept
# of hitting doesn't apply.
nbs = nbs_we_hit + nbs_we_dont

for nb in nbs:
  kernel_user = nb.split('/')[0]
  kernel_slug = nb.split('/')[1]
  full_path = prefix+"/"+nb
  print(f"--- RUNNING: {kernel_user}/{kernel_slug}")
  succ = run_nb.run_nb_paper(full_path, args)
  assert succ
  res = subprocess.run(["mv", "stats.json", f"stats/{kernel_user}_{kernel_slug}.json"])
  assert res.returncode == 0
