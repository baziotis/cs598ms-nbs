import os
import json

import pandas as pd

import tokenize
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim

df_methods = dir(pd.DataFrame)
sr_methods = dir(pd.Series)
pd_methods = dir(pd)

# TODO Configurations that you need to set
path_to_data = r'/home/damitha/ml_data_sys/data/measure_data_gcp/measure_data'
print_issues = True
print_speedups = False
add_conv_time = True

def words2bag(bow, data):
    res = []
    for dat in data:
        re = []
        insts = dat[0]
        size = dat[1]
        deci = dat[2]
        for key in bow.keys():
            if key in insts:
                re.append(1)
            else:
                re.append(0)
        re.append(size)
        re.append(deci)
        # if deci == 0:
        #     re.append(1)
        #     re.append(0)
        # else:
        #     re.append(0)
        #     re.append(1)
        res.append(re)
    return torch.tensor(res).float()

def data_used(instrs, vars):
    res = []
    # drop imports
    for instr in instrs:
        for var in vars:
            if var in instr:
                res.append(var)

    res = list(set(res))
    return res

def data_size(vars, vars_size):
    res = 0
    # drop imports
    for var in vars:
        res += int(vars_size[var])
    return res


def parse_inst(code, vars):
    res = []
    # drop imports
    if any(var in code for var in vars):
        code_bytes = code.encode('utf-8')
        code_bytes_io = BytesIO(code_bytes)

        tokens = tokenize.tokenize(code_bytes_io.readline)
        for tok in tokens:
            if tok.type == tokenize.NAME:
                if (tok.string in df_methods) or (tok.string in sr_methods) or (tok.string in pd_methods):
                    # print(tokenize.tok_name[tok.type], repr(tok.string))
                    res.append(tok.string)

    res = list(set(res))
    return res

def clean_data(data):
    split_data = data.split('\n')

    res = []
    # drop imports
    for x in split_data:
        add_inst = True
        x = x.strip()
        if len(x) == 0:
            add_inst = False
        elif x[0:6] == 'import':
            add_inst = False
        elif x[0:4] == 'from':
            add_inst = False
        elif x[0] == '#':
            add_inst = False

        if (add_inst):
            if '#' in x:
                x = x.split('#')[0]
            res.append(x)
    # print(res)
    return res

nmodin_faster = 0
npandas_faster = 0
sum_speedup_modin = 0
sum_time_modin_fast = 0
sum_time_default = 0
sum_time_pandas_fast = 0
times_modin = []
times_pandas = []
modin_faster = {}

all_instrs = []
fast_instrs = []

gives_spdups = []
no_spdups = []

data_train = []
spdups_train = []

data_test = []
spdups_test = []
times_modin_test = []
times_pandas_test = []
times_modin_no_conv_test = []
sum_time_default_test = 0

working_examples = 0
used_data = 0
skipped_data = 0
nconsec_spdups = 0

test_pds = []

test_pd_name = []

empty_inst_pandas = {}
empty_inst_modin = {}

# test_pds = ['jyotsananegi_melbourne-housing-snapshot-eda.json_1024',
# 'ibtesama_getting-started-with-a-movie-recommendation-system.json_orig',
# 'roopahegde_cryptocurrency-price-correlation.json_256',
# 'carlmcbrideellis_simple-eda-of-kaggle-grandmasters-scheduled.json_256',
# 'muhammadawaistayyab_used-cars-in-pakistan-stats.json_256',
# 'arimishabirin_globalsalary-simple-eda.json_256',
# 'yuliagm_talkingdata-eda-plus-time-patterns.json_1024',
# 'robikscube_big-data-bowl-comprehensive-eda-with-pandas.json_1024',
# 'ampiiere_animal-crossing-villager-popularity-analysis.json_orig',
# 'itzsanju_eda-airline-dataset.json_orig']

# test_pds = ['viviktpharale_house-price-prediction-eda-linear-ridge-lasso.json_256',
# 'vanguarde_h-m-eda-first-look.json_orig',
# 'beratozmen_clash-of-clans-exploratory-data-analysis.json_1024',
# 'carlmcbrideellis_simple-eda-of-kaggle-grandmasters-scheduled.json_256',
# 'muhammadawaistayyab_used-cars-in-pakistan-stats.json_256',
# 'arimishabirin_globalsalary-simple-eda.json_256',
# 'yuliagm_talkingdata-eda-plus-time-patterns.json_1024',
# 'aieducation_what-course-are-you-going-to-take.json_1024',
# 'ampiiere_animal-crossing-villager-popularity-analysis.json_orig',
# 'itzsanju_eda-airline-dataset.json_orig',]


test_pds = ['viviktpharale_house-price-prediction-eda-linear-ridge-lasso.json_orig',
'kimtaehun_simple-preprocessing-for-time-series-prediction.json_256',
'yuliagm_talkingdata-eda-plus-time-patterns.json_1024',
'carlmcbrideellis_simple-eda-of-kaggle-grandmasters-scheduled.json_256',
'muhammadawaistayyab_used-cars-in-pakistan-stats.json_256',
'arimishabirin_globalsalary-simple-eda.json_orig',
'aieducation_what-course-are-you-going-to-take.json_256',
'ampiiere_animal-crossing-villager-popularity-analysis.json_orig',
'itzsanju_eda-airline-dataset.json_256',
'roopacalistus_retail-supermarket-store-analysis.json_1024',]

def read_data(folder_path_modin, folder_path_pandas, suff):
    global print_speedups
    global print_issues
    global add_conv_time

    global nmodin_faster
    global npandas_faster
    global sum_speedup_modin
    global sum_time_modin_fast
    global sum_time_pandas_fast
    global sum_time_default
    global times_modin
    global times_pandas
    global modin_faster
    global all_instrs
    global fast_instrs
    global gives_spdups
    global no_spdups
    global data_train
    global spdups_train

    global data_test
    global spdups_test
    global times_modin_test
    global times_pandas_test
    global times_modin_no_conv_test
    global sum_time_default_test

    global working_examples
    global used_data
    global skipped_data
    global nconsec_spdups

    global test_pds
    global test_pd_name

    global empty_inst_pandas
    global empty_inst_modin

    # Get the list of files in the folder
    file_list = os.listdir(folder_path_modin)

    # Iterate through the files and read each one
    for file_name in file_list:
        if (file_name in [".version", ".ipynb_checkpoints"]):
            continue
        file_path_modin = os.path.join(folder_path_modin, file_name)
        file_path_pandas = os.path.join(folder_path_pandas, file_name)

        conv_times = 0

        with open(file_path_modin, 'r') as file_modin:
            with open(file_path_pandas, 'r') as file_pandas:
                modin_data = json.load(file_modin)
                pandas_data = json.load(file_pandas)

                cells_modin = modin_data["cells"]
                cells_pandas = pandas_data["cells"]
                ncells = len(cells_modin)

                for nc in range(ncells):
                    cell_modin = cells_modin[nc]
                    cell_pandas = cells_pandas[nc]

                    conv_time_p2m = cell_modin['p2m']

                    if (len(conv_time_p2m) > 0):
                        conv_times += 1

        if (conv_times == 0):
            if print_issues:
                print(file_name, "has issues")
            skipped_data += 1
        else:
            used_data += 1
            check_has_spdup = False
            with open(file_path_modin, 'r') as file_modin:
                with open(file_path_pandas, 'r') as file_pandas:
                    modin_data = json.load(file_modin)
                    pandas_data = json.load(file_pandas)

                    cells_modin = modin_data["cells"]
                    cells_pandas = pandas_data["cells"]
                    ncells = len(cells_modin)

                    consec_spdups = False
                    consec_spdups_vars = []

                    pd_name = file_name + "_" + suff

                    empty_inst_modin[pd_name] = 0
                    empty_inst_pandas[pd_name] = 0

                    for nc in range(ncells):
                        cell_modin = cells_modin[nc]
                        cell_pandas = cells_pandas[nc]

                        code = cell_pandas['raw']
                        var_sizes = cell_pandas['vars']
                        vars = list(var_sizes.keys())
                        # print(vars)

                        conv_time_p2m = cell_modin['p2m']
                        conv_time_m2p = cell_modin['m2p']

                        time_modin = cell_modin['total-ns']
                        time_pandas = cell_pandas['total-ns']

                        # print(time_modin, time_pandas)
                        instrs_in_cell = clean_data(code)
                        data_cell = data_used(instrs_in_cell, vars)
                        instr_cell = parse_inst(code, vars)
                        all_instrs += instr_cell

                        size_of_data = float(data_size(data_cell, var_sizes)) / float(1024 * 1024)

                        # print(instrs_in_cell)
                        # print(data_cell)

                        time_modin_w_conv = time_modin

                        if add_conv_time:
                            for c_var in data_cell:
                                if c_var in consec_spdups_vars:
                                    continue
                                else:
                                    consec_spdups_vars.append(c_var)

                                if (c_var in conv_time_m2p.keys()):
                                    time_modin_w_conv += conv_time_m2p[c_var] + conv_time_p2m[c_var]
                                else:
                                    if print_issues:
                                        print("Potential issue in", c_var, "of", file_name)

                        if (time_modin_w_conv < time_pandas) and len(instrs_in_cell) > 0:
                            nmodin_faster += 1
                            sum_speedup_modin += (time_pandas / time_modin_w_conv)
                            # print(instr_cell)

                            # check_has_spdup = True
                            # fast_instrs += instr_cell
                            # sum_time_modin_fast += time_modin_w_conv
                            # if pd_name in test_pds:
                            #     sum_time_default_test += time_pandas
                            #     data_test.append([instr_cell, size_of_data, 1])
                            #     spdups_test.append(time_pandas/time_modin_w_conv)
                            #     times_modin_test.append(time_modin_w_conv)
                            #     times_pandas_test.append(time_pandas)
                            #     test_pd_name.append(pd_name)
                            #     times_modin_no_conv_test.append(time_modin)
                            # else:
                            #     sum_time_default += time_pandas
                            #     data_train.append([instr_cell, size_of_data, 1])
                            #     spdups_train.append(time_pandas/time_modin_w_conv)
                            #     times_modin.append(time_modin_w_conv)
                            #     times_pandas.append(time_pandas)

                            if len(instr_cell) > 0:
                                check_has_spdup = True
                                fast_instrs += instr_cell
                                sum_time_modin_fast += time_modin_w_conv
                                if pd_name in test_pds:
                                    sum_time_default_test += time_pandas
                                    data_test.append([instr_cell, size_of_data, 1])
                                    spdups_test.append(time_pandas / time_modin_w_conv)
                                    times_modin_test.append(time_modin_w_conv)
                                    times_pandas_test.append(time_pandas)
                                    test_pd_name.append(pd_name)
                                    times_modin_no_conv_test.append(time_modin)
                                else:
                                    sum_time_default += time_pandas
                                    data_train.append([instr_cell, size_of_data, 1])
                                    spdups_train.append(time_pandas / time_modin_w_conv)
                                    times_modin.append(time_modin_w_conv)
                                    times_pandas.append(time_pandas)

                            else:
                                empty_inst_modin[pd_name] += time_modin
                                empty_inst_pandas[pd_name] += time_pandas
                                print("Potential issue in", file_name, instrs_in_cell)

                            if consec_spdups:
                                nconsec_spdups += 1
                            consec_spdups = True

                            if print_speedups:
                                print(file_name)
                                print(code)
                        elif len(instrs_in_cell) > 0:
                            if len(instr_cell) > 0:
                                sum_time_pandas_fast += time_pandas

                                if pd_name in test_pds:
                                    sum_time_default_test += time_pandas
                                    data_test.append([instr_cell, size_of_data, 0])
                                    spdups_test.append(time_pandas / time_modin_w_conv)
                                    times_modin_test.append(time_modin_w_conv)
                                    times_pandas_test.append(time_pandas)
                                    test_pd_name.append(pd_name)
                                    times_modin_no_conv_test.append(time_modin)
                                else:
                                    sum_time_default += time_pandas
                                    data_train.append([instr_cell, size_of_data, 0])
                                    spdups_train.append(time_pandas / time_modin_w_conv)
                                    times_modin.append(time_modin_w_conv)
                                    times_pandas.append(time_pandas)

                            npandas_faster += 1
                            consec_spdups = False
                            consec_spdups_vars = []
                    if check_has_spdup:
                        gives_spdups.append(pd_name)
                    else:
                        no_spdups.append(pd_name)
                    # break
                # print(file_name)

folder_path_modin_loc = path_to_data + '/Original/Modin'
folder_path_pandas_loc = path_to_data + '/Original/Pandas'
suff_loc = "orig"
read_data(folder_path_modin_loc, folder_path_pandas_loc, suff_loc)

folder_path_modin_loc = path_to_data + '/256MB/Modin'
folder_path_pandas_loc = path_to_data + '/256MB/Pandas'
suff_loc = "256"
read_data(folder_path_modin_loc, folder_path_pandas_loc, suff_loc)

folder_path_modin_loc = path_to_data + '/1GB/Modin'
folder_path_pandas_loc = path_to_data + '/1GB/Pandas'
suff_loc = "1024"
read_data(folder_path_modin_loc, folder_path_pandas_loc, suff_loc)

all_instrs = list(set(all_instrs))

bow_instr = {}
i = 0
for inst in all_instrs:
    bow_instr[inst] = i
    i += 1

input_size = len(list(set(all_instrs))) + 1

#------------------------------- Models ------------------------
#------------------------------- Simple NN ------------------------

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define input size, hidden size, and output size
# input_size = 10  # Example input size
hidden_size = 64  # Example hidden layer size
output_size = 2  # Example output size

# Create an instance of the neural network
model = SimpleNN(input_size, hidden_size, output_size)

# Define a sample dataset and corresponding labels
# Replace this with your actual dataset
inputs = words2bag(bow_instr, data_train)
inputs_test = words2bag(bow_instr, data_test)
print(inputs.shape)

labels = inputs[:, -1].type(torch.LongTensor)
labels_test = inputs_test[:, -1].type(torch.LongTensor)
# labels = torch.unsqueeze(labels, 1)
print(labels.shape)
inputs = inputs[:, :-1].float()
inputs_test = inputs_test[:, :-1].float()
print(inputs.shape)
# torch.randint(0, output_size, (100,))  # Example labels (0, 1, or 2) for 100 samples

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1600

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Print results
correct = 0
all = 0

for x in range(len(outputs)):
    label = labels[x].item()
    if (label == 0):
        if outputs[x][0].item() > outputs[x][1].item():
            correct+= 1
    else:
        if outputs[x][0].item() < outputs[x][1].item():
            # print("is here")
            correct+= 1
    all += 1

print(correct, all, correct/all)
outputs = model(inputs_test)
correct = 0
all = 0
total_spdups = 0

total_deci_time = 0

for d1 in range(len(outputs)):
    l1 = labels_test[d1]
    # print(predictions[d1])
    res_loc = outputs[d1][0] > outputs[d1][1]
    if (l1 == 0):
        if (res_loc):
            total_spdups += 1
            correct+= 1
            total_deci_time += times_pandas_test[d1]
        else:
            total_spdups += spdups_test[d1]
            total_deci_time += times_modin_test[d1]
    else:
        if not(res_loc):
            # print("is here")
            total_spdups += spdups_test[d1]
            correct += 1
            total_deci_time += times_modin_test[d1]
        else:
            total_spdups += 1
            total_deci_time += times_pandas_test[d1]
    all += 1
print("-------- Results for NN - Test ------")
print("correct, all, accuracy, total speedups")
print(correct, all, correct/all, sum_time_default_test/total_deci_time)
print("--------------")

#------------------------------- XGBoost Model --------------------------------
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Generate a sample classification dataset
X = inputs.numpy()
y = labels.numpy()

X_test = inputs_test.numpy()
y_test = labels_test.numpy()


# Create an XGBoost classifier and fit it to the training data
model = xgb.XGBClassifier()

# Best for with no conversion costs
# model = xgb.XGBClassifier(
#     tree_method='gpu_hist',
#     booster='dart',
#     objective='binary:hinge',
#     random_state=42,
#     learning_rate=0.01,
#     # colsample_bytree=0.9,
#     # colsample_bylevel=0.3,
#     # colsample_bynode=0.9,
#     max_depth=30, # increased to 15
#     n_estimators=200,
#     enable_categorical=True
#     )

model.fit(X, y)

# Make predictions on the test set
predictions = model.predict(X)
predictions_test = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y, predictions)
print(f"Accuracy: {accuracy:.4f}")

accuracy = accuracy_score(y_test, predictions_test)
print(f"Accuracy: {accuracy:.4f}")

# Print results

correct = 0
all = 0
total_spdups = 0

total_deci_time = 0

for d1 in range(len(predictions)):
    l1 = y[d1]
    # print(predictions[d1])
    if (l1 == 0):
        if (predictions[d1] == 0):
            total_spdups += 1
            correct+= 1
            total_deci_time += times_pandas[d1]
        else:
            total_spdups += spdups_train[d1]
            total_deci_time += times_modin[d1]
    else:
        if (predictions[d1] == 1):
            # print("is here")
            total_spdups += spdups_train[d1]
            correct += 1
            total_deci_time += times_modin[d1]
        else:
            total_spdups += 1
            total_deci_time += times_pandas[d1]
    all += 1

#%%
print(correct, all, correct/all, total_spdups/all, sum_time_default_test/total_deci_time)

correct = 0
all = 0
total_spdups = 0
has_speedup = 0

total_deci_time = 0
total_best_time = 0
total_def_time = 0

curr_pd_name = test_pd_name[0]
pandas_time = empty_inst_pandas[curr_pd_name]
modin_time = empty_inst_modin[curr_pd_name]
model_time = empty_inst_pandas[curr_pd_name]

print("Detailed decisions by the model")

for d1 in range(len(predictions_test)):
    new_name = test_pd_name[d1]

    if (curr_pd_name != new_name):
        print(pandas_time / 1000000000, modin_time/ 1000000000, model_time/ 1000000000, "for", curr_pd_name)
        pandas_time = empty_inst_pandas[curr_pd_name]
        modin_time = empty_inst_modin[curr_pd_name]
        model_time = empty_inst_pandas[curr_pd_name]
        curr_pd_name = new_name

    pandas_time += times_pandas_test[d1]
    modin_time += times_modin_no_conv_test[d1]
    l1 = y_test[d1]
    total_def_time += times_pandas_test[d1]
    # print(predictions[d1])
    if (l1 == 0):
        total_best_time += times_pandas_test[d1]
        if (predictions_test[d1] == 0):
            total_spdups += 1
            correct+= 1
            total_deci_time += times_pandas_test[d1]
            model_time += times_pandas_test[d1]
        else:
            print("is here - bad - add slowdown")
            total_spdups += spdups_test[d1]
            total_deci_time += times_modin_test[d1]
            model_time += times_modin_test[d1]
    else:
        total_best_time += times_modin_test[d1]
        has_speedup += 1
        # print("is here")
        if (predictions_test[d1] == 1):
            print("is here - good")
            total_spdups += spdups_test[d1]
            correct += 1
            total_deci_time += times_modin_test[d1]
            model_time += times_modin_test[d1]
        else:
            print("is here - bad - miss speedup")
            total_spdups += 1
            total_deci_time += times_pandas_test[d1]
            model_time += times_pandas_test[d1]
    all += 1
print(pandas_time / 1000000000, modin_time/ 1000000000, model_time/ 1000000000, "for", curr_pd_name)

print("-------- Results for XGBoost - Test ------")
print("correct, all, accuracy, total speedups from model, possible speedups")
print(correct, all, correct/all, sum_time_default_test/total_deci_time, sum_time_default_test/total_best_time)
print("--------------")
# print(correct, all, has_speedup, correct/all, total_spdups/all, sum_time_default_test/total_deci_time, sum_time_default_test/total_best_time)