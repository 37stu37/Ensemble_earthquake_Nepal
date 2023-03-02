
from matplotlib import pyplot as plt


def plot_AUCROC(y_true, y_model):
    from sklearn import metrics

    # calculate AUC of model
    score = metrics.roc_auc_score(y_true, y_model)

    # print AUC score
    print(f"ROC auc score => {score}")

    false_positive_rate, true_positive_rate, thr = metrics.roc_curve(y_true, y_model)

    plt.figure(dpi=100) # figsize=(5, 4), dpi=150
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='red', alpha=0.7)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.text(0.95, 0.05, 'AUC-ROC = %0.2f' % score, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    return


def plot_AUCF1(y_true, y_model):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import PrecisionRecallDisplay
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc

    prec, recall, _ = precision_recall_curve(y_true, y_model)
    # pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, estimator_name='pr_auc')
    # f1 = f1_score(y_true, y_model, average="macro")
    print(f"prec/recall auc score => {auc(recall, prec)}")
    score = auc(recall, prec)

    plt.figure(dpi=100) # figsize=(5, 4), dpi=150
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("F1 curve")
    plt.plot(recall, prec, 'b')
    plt.fill_between(recall, prec, facecolor='red', alpha=0.7)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.text(0.95, 0.05, 'AUC-F1 = %0.2f' % score, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    return


def plot_qc_f1_auc_threshold(Impact_DF, scale):

    """
    at each scale create a plot showing the f1 and recall scores with varying thresholds
    added also the r square for continuous data

    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from sklearn.metrics import r2_score
    import numpy as np
    import pandas as pd

    # Get the r square score
    grpby = Impact_DF.groupby(scale).agg({'impact': 'sum', 'E4': 'sum'}).reset_index()
    model = grpby.impact.values
    actual = grpby.E4.values

    r2=r2_score(actual, model)

    # Get the roc and f1 score
    grpby = Impact_DF.groupby(scale).agg({'impact': 'sum', 'E4': 'max'}).reset_index()
    model = grpby.impact.values
    # model = preprocessing.normalize(model.reshape(1, -1))
    actual = grpby.E4.values

    interval = max(model)/100
    # interval = np.rint(max(model)).astype(int) / 50
    l_threshold = np.arange(0, max(model), interval)

    # print(f"scale = {s cales[i]}")
    f1_list=[]
    a_list=[]
    recall_list=[]
    precision_list=[]
    tp_list = []
    fp_list = []
    fn_list = []
    tn_list = []
    
    for l in range(len(l_threshold)):
        # print(l_threshold[l])
        confM = confusion_matrix(actual, np.where(model>l_threshold[l], 1, 0))
        tp = confM[1,1]
        fp = confM[0,1]
        fn = confM[1,0]
        tn = confM[0,0]
        
        PR = tp / (tp + fp)
        RC = tp / (tp + fn)
        recall_list.append(RC)
        precision_list.append(PR)

        fpr, tpr, _ = roc_curve(actual, np.where(model>l_threshold[l], 1, 0))
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(actual, np.where(model>l_threshold[l], 1, 0), average="macro")
        f1_list.append(f1)
        a_list.append(roc_auc)
        
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        tn_list.append(tn)

    l_threshold = l_threshold / np.max(l_threshold)
    
    thresholds_results = pd.DataFrame(
    {'thresholds': l_threshold,
     'f1': f1_list,
     'roc': a_list,
     'tp' : tp_list,
     'fp' : fp_list,
     'fn' : fn_list,
     'tn' : tn_list
    })

    return thresholds_results, r2


def plot_typology_fragility(typo, l_mean, l_scale):
    from scipy.stats import lognorm
    import matplotlib.pyplot as plt
    import numpy as np

    typology=typo
    alf = 1.0

    pgas = np.linspace(0,3, 100)

    h = [lognorm(l_mean[0],scale=l_scale[0]).cdf(p) for p in pgas]
    m = [lognorm(l_mean[1],scale=l_scale[1]).cdf(p) for p in pgas]
    l = [lognorm(l_mean[2],scale=l_scale[2]).cdf(p) for p in pgas]

    fig, axs = plt.subplots(1, 1, figsize=(5,3))

    axs.plot(pgas, l, c='lightblue', label="low case", alpha=alf)
    axs.plot(pgas, m, c='lightgreen', label="mid case", alpha=alf)
    axs.plot(pgas, h, c='red', label="high case", alpha=alf)

    axs.title.set_text(f'fragility function for {typology}')
    axs.set(ylabel="Probability of collapse", xlabel="PGA (g)")
    axs.set_ylim(0,1.1)
    axs.legend()
    axs.legend(loc='lower right', frameon=True)
    plt.show()

    return