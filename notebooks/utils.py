import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def summary(df):
    summary_df = pd.DataFrame({'variable':df.columns})
    # visualize dtype
    summary_df['dtype'] = summary_df.variable.apply(lambda x: df[x].dtype) 
    # analyze unique values to know cardinality of variables
    summary_df['unique'] = summary_df.variable.apply(lambda x: len(df[x].unique()))
    # see if there are empty cells
    summary_df['empty'] = summary_df.variable.apply(lambda x: len([e for e in df[x] 
                                                                   if (str(e).isspace() or 
                                                                       len(str(e))==0)]))
    # analyze presence of missing or null values
    summary_df['na'] = summary_df.variable.apply(lambda x: df[x].isna().sum())
    # calculate missing percentage
    summary_df['%_na'] = summary_df.variable.apply(lambda x: round(df[x].isna().sum()/len(df)*100,1))
    summary_df['null'] = summary_df.variable.apply(lambda x: df[x].isnull().sum())
    summary_df['%_null'] = summary_df.variable.apply(lambda x:round(df[x].isnull().sum()/len(df)*100,1))
    # compute number of zeros
    summary_df['zeros'] = summary_df.variable.apply(lambda x: len(df[df[x]==0]))
    summary_df['%_zeros'] = summary_df.variable.apply(lambda x:round(len(df[df[x]==0])/len(df)*100,1))
    return summary_df


def draw_histograms(df, variables):
    n_cols = 5
    n_rows = round(len(variables)/n_cols) + 1
    fig=plt.figure(figsize=(15,15))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout() # improves layout
    plt.show()
    
    
def make_corr_plot(df):
    fig = plt.figure(figsize=(12,12))
    data = df.corr()
    ax = sns.heatmap(data,
                     vmin=-1, vmax=1, center=0,
                     cmap=sns.diverging_palette(20, 220, n=10),
                     square=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
    
    
def draw_boxplots_target(df, variables, target):
    for var in variables:
        sns.boxplot(x=df[target], y=df[var],
                    showfliers = False)
        plt.title(f'{var} distribution by target')
        plt.show()
        

def draw_catplots_target(df, variables, target):
    for var in variables:
        list_ = list(df[var].value_counts()[:10].index)
        sns.catplot(
        data=df[df[var].isin(list_)], y=var, hue=target, kind="count",
        palette="pastel", edgecolor=".6",
        )   
        plt.title(f'Top (10) categories {var}')
        plt.show()
        
# CONFUSION MATRIX
def create_confusion_matrix_binary(y_real, y_pred, cat_labels=[0, 1], title="Confusion Matrix", 
                                   cmap="Blues"):
    '''
    Creates a confusion matrix for a binary classifaction model.

    Takes the following input:
    - y_real = an array or pandas serie with the real categories
    - y_pred = an array or pandas serie with the predicted categories
    - (optional) cat_labels = a list with the names of the two categories
    - (optional) title = title of the confusion matrix
    - (optional) cmap = color palette
    '''
    
    cm = confusion_matrix(y_real, y_pred)

    group_names = ["True Negative","False Positive","False Negative","True Positive"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt="", cmap=cmap)
    plt.title(title, fontsize=14, y=1.05)
    plt.xlabel("predictions", fontsize=12)
    plt.xticks(ticks=[0.5,1.5], labels=cat_labels, fontsize=12)
    plt.ylabel("real values", fontsize=12)
    plt.yticks(ticks=[0.5,1.5], labels=cat_labels, fontsize=12)
    
    plt.show()

def choose_model(model_metrics, metric_to_max='auc', min_diff=0.02, max_2nd_metric='recall'):
    """
    Choose a model based on specified criteria.
    
    Parameters:
        model_metrics (dict): Dictionary containing metrics for each model.
        metric_to_max (str): Metric to use for comparison. Default is 'auc'.
        min_metric_diff (float): Minimum difference in metric required to choose a model. Default is 0.02.
        max_2nd_metric (str): Second Metric to maximize in case of first metric difference is lower than threshold.
        
    Returns:
        str: Name of the chosen model.
    """
    # Find the model with the highest AUC
    best_model = max(model_metrics.items(), key=lambda x: x[1][metric_to_max])
    for model, metrics in model_metrics.items():
        if metrics[metric_to_max] - best_model[1][metric_to_max] >= min_diff:
            print(f'Model selected maximizing {metric_to_max}')
            return model
        else:
            # Find the model with the highest recall
            best_model = max(model_metrics.items(), key=lambda x: x[1][max_2nd_metric])
            print(f'Model selected maximizing {max_2nd_metric}')
            return best_model[0]
        
def extract_metric(report_str, metric, target_class=1):
    """
    Extract a specific metric value for a target class from sklearn classification report.

    Parameters:
        report_str (str): String containing the classification report.
        metric (str): Metric to extract (e.g., 'precision', 'recall', 'f1-score').
        target_class (int): Target class for which to extract the metric. Default is 1.

    Returns:
        float: Metric value for the target class.
    """
    lines = report_str.split('\n')
    for line in lines:
        if line.strip().startswith(str(target_class)):
            metric_value = float(line.split()[list(report_str.split('\n')[0].split()).index(metric) + 1])
            return metric_value
        
def plot_log_loss(cutoffs, y, y_pred):
    log_loss = []
    for c in cutoffs:
        log_loss.append(
            metrics.log_loss(y, np.where(y_pred > c, 1, 0))
        )
    plt.figure(figsize=(15,10))
    plt.plot(cutoffs, log_loss)
    plt.xlabel("Cutoff")
    plt.ylabel("Log loss")
    plt.show()
    return log_loss