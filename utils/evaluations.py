import os
import csv
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import time
#from prg import prg
# import cv2
sns.set(color_codes=True)


def my_confusion_matrix(y, y_pred):
    TN = 0
    FN = 0
    FP = 0
    TP = 0
    for i in range(y.shape[0]):
        if y[i]==0 and y_pred[i]==0:
            TN = TN+1
        elif y[i]==1 and y_pred[i]==0:
            FN = FN+1
        elif y[i]==0 and y_pred[i]==1:
            FP = FP+1
        elif y[i]==1 and y_pred[i]==1:
            TP = TP+1

    return TN, FN, FP, TP

def derive_metric(TN, FN, FP, TP):
    overall = float(TP+TN)/float(TP+TN+FN+FP)
    average = (TP/float(TP+FP)+TN/float(FN+TN))/2
    sens = TP/float(TP+FN)
    spec = TN/float(TN+FP)
    ppr = TP/float(TP+FP)

    return overall, average, sens, spec, ppr


def do_roc(scores, true_labels, file_name='', directory='', plot=False):
    """ Does the ROC curve

    Args:
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the ROC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            roc_auc (float): area under the under the ROC curve
            thresholds (list): list of thresholds for the ROC
    """
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr) # compute area under the curve
    if plot: 
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        plt.savefig(directory + file_name + 'roc.png')
        plt.close()

    return roc_auc

def do_prc(scores, true_labels, file_name='', directory='', plot=False):
    """ Does the PRC curve

    Args :
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the PRC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            prc_auc (float): area under the under the PRC curve
            pre_score (float): precision score
            rec_score (float): recall score
            F1_score (float): max F1 score according to different thresholds
    """
    precision, recall, _ = precision_recall_curve(true_labels, scores)

    prc_auc = auc(recall, precision)

    if plot:
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AUC=%0.4f'
                            %(prc_auc))
        plt.savefig(directory + file_name + 'prc.png')
        plt.close()

    return prc_auc

def do_prg(scores, true_labels, file_name='', directory='', plot=False):
    prg_curve = prg.create_prg_curve(true_labels, scores)
    auprg = prg.calc_auprg(prg_curve)
    if plot:
       prg.plot_prg(prg_curve)
       plt.title('Precision-Recall-Gain curve: AUC=%0.4f'
                            %(auprg))
       plt.savefig(directory + file_name + "prg.png")


    return auprg

def do_cumdist(scores, file_name='', directory='', plot=False):
    N = len(scores)
    X2 = np.sort(scores)
    F2 = np.array(range(N))/float(N)
    if plot:
        plt.figure()
        plt.xlabel("Anomaly score")
        plt.ylabel("Percentage")
        plt.title("Cumulative distribution function of the anomaly score")
        plt.plot(X2, F2)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + file_name + 'cum_dist.png')

def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([28,28])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

def computeTSNEProjectionOfLatentSpace(X, latent, name, display=False):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = latent

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    #tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    algo = manifold.Isomap(n_neighbors=5, n_components=2)
    X_tsne = algo.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    print("Plotting t-SNE visualization...")
    fig, ax = plt.subplots()
    imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.6)
    if display:
        plt.show()
    else:
        plt.savefig("results/tsne/"+name+"_tsne.png")

def predict(scores, threshold):
    return scores>=threshold

def make_meshgrid(x_min,x_max,y_min,y_max, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def save_grid_plot(samples, samples_rec, name_model, dataset, nb_images=50,
                   grid_width=10):

    args = name_model.split('/')[:-1]
    directory = os.path.join(*args)
    if not os.path.exists(directory):
        os.makedirs(directory)
    samples = (samples + 1) / 2
    samples_rec = (samples_rec + 1) / 2
    if dataset == 'mnist':
        figsize = (28,28)
    elif dataset == 'rop':
        figsize = (128,128)
    else:
        figsize = (32, 32)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_width, grid_width)
    gs.update(wspace=0.05, hspace=0.05)
    list_samples = []
    for x, x_rec in zip(np.split(samples, nb_images // grid_width),
                        np.split(samples_rec, nb_images // grid_width)):
        list_samples += np.split(x, grid_width) + np.split(x_rec, grid_width)
    list_samples = [np.squeeze(sample) for sample in list_samples]
    for i, sample in enumerate(list_samples):
        if i>=nb_images:
            break
        ax = plt.subplot(gs[i])
        plt.imshow(sample)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig('{}.png'.format(name_model))

def save_results(scores, true_labels, model, dataset, method, weight, label,
                 random_seed, anomaly_type, anomaly_proportion, step=-1):

    directory = 'results/{}/{}/{}_{}/{}/w{}/'.format(model,
                                                  dataset,
                                                  anomaly_type,
                                                  anomaly_proportion,
                                                  method,
                                                  weight)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(directory, dataset)
    if dataset != 'kdd':

        print("Tets on ", dataset)
        file_name = str(label)+"_step"+str(step)
        if anomaly_type == 'novelty':
            print("NOVELTY")
            c = 90
            if dataset == 'rop':
                c = 22
        else:
            c = anomaly_proportion * 100

        file_name = "{}_step{}_rd{}".format(label, step, random_seed)
        c = anomaly_proportion * 100

        # Highest 5% are anomalous
        per = np.percentile(scores, 100 - c)
        fname = directory + "{}.csv".format(label)
        csv_file = directory + "scores.csv"
    else:
        file_name = "kdd_step{}_rd{}".format(step, random_seed)
        # Highest 20% are anomalous
        per = np.percentile(scores, 80)
        fname = directory + "results.csv"
        csv_file = directory + "scores.csv"
    
    scores = np.array(scores) 


    csv = pd.DataFrame()
    csv['scores']=scores
    csv['labels']=true_labels
    csv.to_csv(csv_file, index=False)



    #try:
    #    scores_norm = (scores-min(scores))/(max(scores)-min(scores))
    #except:
    #    scores_norm = (scores-scores.min())/(scores.max()-scores.min())
        
    print(max(scores), min(scores))
    roc_auc = do_roc(scores, true_labels, file_name=file_name,
                    directory=directory)
    prc_auc = do_prc(scores, true_labels, file_name=file_name,
                        directory=directory)
    do_cumdist(scores, file_name=file_name, directory=directory)

    
    prg_auc = 0#do_prg(scores, true_labels, file_name=file_name, directory=directory)
    '''
    plt.close()

    plt.figure()
    idx_inliers = true_labels == 0
    idx_outliers = true_labels == 1
    hrange = (min(scores), max(scores))
    plt.hist(scores[idx_inliers], 50, facecolor=(0, 1, 0, 0.5),
             label="Normal samples", density=True, range=hrange)
    plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),
             label="Anomalous samples", density=True, range=hrange)
    plt.title("Distribution of the anomaly score")
    plt.legend()
    plt.savefig(directory + 'histogram_{}_{}.png'.format(random_seed, dataset),
                transparent=True, bbox_inches='tight')
    '''

    y_pred = (scores>=per)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels.astype(int),
                                                               y_pred.astype(int),
                                                               average='binary')

    print("Testing at step %i, method %s: Prec = %.4f | Rec = %.4f | F1 = %.4f"
        % (step, method, precision, recall, f1))

    print("Testing method {} | ROC AUC = {:.4f} | PRC AUC = {:.4f} | PRG AUC = {:.4f}".format(method, roc_auc,
                                                                                              prc_auc, prg_auc))

    results = [model, dataset, anomaly_type, anomaly_proportion, method, weight, label,
               step, roc_auc, prc_auc, prg_auc, precision, recall, f1, random_seed, time.ctime()]
    save_results_csv("results/results.csv", results, header=0)
    
    results = [step, roc_auc, prc_auc, precision, recall, f1, random_seed]
    save_results_csv(fname, results, header=0)

def save_results_csv(fname, results, header=0):
    """Saves results in csv file
    Args:
        fname (str): name of the file
        results (list): list of prec, rec, F1, rds
    """

    new_rows = []
    if not os.path.isfile(fname):
        args = fname.split('/')[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(fname, 'wt') as f:
            writer = csv.writer(f)
            if header == 0:
                writer.writerows(
                    [['Model', 'Dataset', 'AnomalyType', 'AnomalyProportion', 'Method', 'Weight', 'Label', 
                      'Step', 'AUROC', 'AUPRC', 'AUPRG', 'Precision', 'Recall',
                      'F1 score', 'Random Seed', 'Date']])
            if header == 1:
                writer.writerows(
                    [['Precision', 'Recall', 'F1 score', 'Random Seed']])
            elif header ==2:
                writer.writerows(
                    [['Step', 'AUROC', 'AUPRC', 'Precision', 'Recall',
                      'F1 score', 'Random Seed']])
            elif header ==3:
                writer.writerows(
                    [['Inception score generated', 'Inception score reconstructed',
                      'FID generated', 'FID reconstructed',
                      'Epoch']])

    with open(fname, 'rt') as f:
        reader = csv.reader(f)  # pass the file to our csv reader
        for row in reader:  # iterate over the rows in the file
            new_rows.append(row)

    with open(fname, 'wt') as f:
        # Overwrite the old file with the modified rows
        writer = csv.writer(f)
        new_rows.append(results)  # add the modified rows
        writer.writerows(new_rows)

def heatmap(data, name=None, save=False):

    fig = plt.figure()
    ax = sns.heatmap(data, cmap="YlGnBu")
    fig.add_subplot(ax)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save:
        args = name.split('/')[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('{}.png'.format(name))

    return data

def clustermap(scores, labels, name=None, save=True):

    fig = plt.figure()
    d = {'Scores': pd.Series(scores),
         '': pd.Series(scores),
         'Labels': pd.Series(labels)}
    df = pd.DataFrame(d)
    labels = df.pop("Labels")
    lut = dict(zip(labels.unique(), "rg"))
    row_colors = labels.map(lut)
    g = sns.clustermap(df, cmap="mako", row_colors=row_colors)
    if save:
        args = name.split('/')[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('{}.png'.format(name))
