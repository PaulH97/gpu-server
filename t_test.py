from scipy.stats import t
from math import sqrt
from statistics import stdev
import yaml
import os
from tools_model import load_trainData
import numpy as np

def corrected_t_test(y1, y2, alpha=0.05):
    
    # Calculate the differences between y1 and y2
    y1 = np.array(y1)
    y2 = np.array(y2)
    d = y1 - y2
    print(d)
    # Calculate the mean difference
    mean_d = np.mean(d)
    # Calculate the standard deviation of the differences
    s_d = np.std(d, ddof=1)
    # Calculate the number of differences
    n = len(d)
    # Calculate the standard error of the mean difference
    se_d = s_d / np.sqrt(n)
    # Calculate the t-statistic
    t_stat = mean_d / se_d
    # Calculate the degrees of freedom
    df = n - 1
    # Calculate the critical value
    cv = t.ppf(1 - alpha/2, df)
    # Calculate the confidence interval
    ci = round(mean_d - cv * se_d, 3), round(mean_d + cv * se_d, 3)
    # Calculate the p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    # Determine if the difference is statistically significant
    if p_value < alpha:
        significance = True
    else:
        significance = False
    # Return the results
    return round(mean_d,3), round(s_d,3), round(se_d,3), round(t_stat,3), df, round(p_value,3), significance, ci

def getMetrics(model_path):
    
    metric_file = os.path.join(model_path, "metrics.txt")
    metric_file = open(metric_file, "r")
    metric_lines = metric_file.readlines()

    metrics_folds = [line.strip() for line in metric_lines if line[0] == "{"]
    
    recall = [eval(fold)["recall"] for fold in metrics_folds]
    precision = [eval(fold)["precision"] for fold in metrics_folds]
    iou = [eval(fold)["iou"] for fold in metrics_folds]
    dice = [eval(fold)["dice_metric"] for fold in metrics_folds]
    f1 = [(2*pr[0]*pr[1])/(pr[0] + pr[1]) for pr in zip(precision,recall)]  
    
    all_metrics = np.array([precision, recall, f1, iou, dice]) *100
    
    return all_metrics

model1 = "/home/hoehn/data/output/Sentinel-2/models2/S2_256idx_kfold"
model2 = "/home/hoehn/data/output/Sentinel-12/models2/S12_256idx_kfold"

textfile = "S2_256idx_vs_S12_256idx.txt"

metrics = ["Precision", "Recall", "F1-score", "iou", "dice_metric"]
m1_metrics = getMetrics(model1)
m2_metrics = getMetrics(model2)

# Perform corrected paired Student's t-tests for each metric
results = []
for i, metric in enumerate(metrics):
    mean_d, s_d, se_d, t_stat, df, p_value, significance, ci = corrected_t_test(m1_metrics[i], m2_metrics[i])
    result = {"Metric": metric, "Mean Difference": mean_d, "Standard Deviation of Differences": s_d,
              "Standard Error of Mean Difference": se_d, "T-Statistic": t_stat, "Degrees of Freedom": df,
              "P-Value": p_value, "Significance": significance, "Confidence Interval": ci}
    results.append(result)

out_text = os.path.join("/home/hoehn/data/t_test/sentinel", textfile)

# Write results to a text file
with open(out_text, "w") as f:
    f.write("Results of Corrected Paired Student's t-Tests for model {} and {}:\n".format(model1.split("/")[-1], model2.split("/")[-1]))
    for result in results:
        f.write("-----------\n")
        for key, value in result.items():
            f.write(f"{key}: {value}\n")
