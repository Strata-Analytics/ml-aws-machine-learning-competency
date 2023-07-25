import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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