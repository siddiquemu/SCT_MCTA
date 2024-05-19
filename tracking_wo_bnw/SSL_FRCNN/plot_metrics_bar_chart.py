#######################################
#
# Script for plotting tracking performance of MRCNN+MHT using Bar diagram
#
# Tracking Results from MHT: %MT = MT/GT, PT?? : should be - (MT+PT)/GT
#######################################
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter


"""
9:clasp1 - P-Base
Average Precision: 0.6815
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
76.9  72.5  81.8| 0.04    575    417     93    158| 56.3  81.1  

9:clasp1 - B-Base 
Average Precision: 0.3302
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
50.7  34.8  93.7| 0.00    299    104      7    195| 32.4  81.6  

11:clasp1 - P-Base
Average Precision: 0.7521
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
84.4  75.0  96.5| 0.01    625    469     17    156| 72.3  83.1 

11:clasp1 - B-Base
Average Precision: 0.3488
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
51.7  37.4  83.8| 0.01    345    129     25    216| 30.1  78.9
--------------------------------------------------------------

9:clasp1 - P-SL
Average Precision: 0.9487
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
96.4  95.7  97.2| 0.01    575    550     16     25| 92.9  85.5 

9:clasp1 - B-SL
Average Precision: 0.8728
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
90.1  90.0  90.3| 0.01    299    269     29     30| 80.3  84.4 

11:clasp1 - P-SL
Average Precision: 0.9745
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
98.2  97.6  98.9| 0.00    625    610      7     15| 96.5  86.3  

11:clasp1 - B-SL
Average Precision: 0.8742
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
90.7  89.0  92.5| 0.01    345    307     25     38| 81.7  82.2  

---------------------------------------------------------------
9:clasp1 - P-SSL


9:clasp1 - B-SSL
Average Precision: 0.7749
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
84.1  77.6  91.7| 0.01    299    232     21     67| 70.6  79.2 

11:clasp1-P-SSL
 

11:clasp1 - B-SSL
Average Precision: 0.7240
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
81.6  74.8  89.9| 0.01    345    258     29     87| 66.4  77.2  

---------------------------------------------------------------
9:clasp1 - P-SSL-alpha
Average Precision: 0.9461
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
95.9  96.2  95.7| 0.01    575    553     25     22| 91.8  83.8

9:clasp1 - B-SSL-alpha
Average Precision: 0.8000
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
86.7  80.6  93.8| 0.01    299    241     16     58| 75.3  80.3 

11:clasp1-P-SSL-alpha
 Average Precision: 0.9462
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
98.1  97.4  98.9| 0.00    625    609      7     16| 96.3  85.0

11:clasp1 - B-SSL-alpha
Average Precision: 0.7240
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
81.6  74.8  89.9| 0.01    345    258     29     87| 66.4  77.2  
-------------------------------------------------------------------------

CLASP2 Datasets:
---------------
9:clasp2 - P-Base
Average Precision: 0.7314
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
79.8  75.8  84.3| 0.00    715    542    101    173| 61.7  83.5

9:clasp2 - B-Base 
Average Precision: 0.4061
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
56.3  43.9  78.5| 0.00    367    161     44    206| 31.9  75.1

11:clasp2 - P-Base
Average Precision: 0.6454
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
76.3  67.0  88.6| 0.00    197    132     17     65| 58.4  83.2 

11:clasp2 - B-Base
Average Precision: 0.1741
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
34.8  22.5  77.5| 0.00    138     31      9    107| 15.9  79.8 
--------------------------------------------------------------
9:clasp2 - P-SL
Average Precision: 0.9225
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
94.7  94.5  94.8| 0.00    715    676     37     39| 89.4  85.9 

9:clasp2 - B-SL
Average Precision: 0.9479
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
95.1  95.6  94.6| 0.00    367    351     20     16| 90.2  85.4 

11:clasp2 - P-SL
Average Precision: 0.9023
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
95.0  92.4  97.8| 0.00    197    182      4     15| 90.4  84.6  

11:clasp2 - B-SL
Average Precision: 0.8825
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
90.6  90.6  90.6| 0.00    138    125     13     13| 81.2  84.5 

---------------------------------------------------------------
9:clasp2 - P-SSL
Average Precision: 0.8974
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
92.8  90.8  95.0| 0.00    715    649     34     66| 86.0  83.0 

9:clasp2 - B-SSL
Average Precision: 0.6993
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
82.6  72.5  96.0| 0.00    367    266     11    101| 69.5  76.9  

11:clasp2-P-SSL
Average Precision: 0.9225
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
93.6  93.4  93.9| 0.00    197    184     12     13| 87.3  82.7  

11:clasp2 - B-SSL
Average Precision: 0.5766
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
72.2  59.4  92.1| 0.00    138     82      7     56| 54.3  80.3 

---------------------------------------------------------------
9:clasp2 - P-SSL-alpha
Average Precision: 0.9179
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
94.2  94.5  93.9| 0.00    715    676     44     39| 88.4  82.4 

9:clasp2 - B-SSL-alpha
Average Precision: 0.8446
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
88.8  86.6  91.1| 0.00    367    318     31     49| 78.2  78.3

11:clasp2-P-SSL-alpha
Average Precision: 0.9261
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
96.6  94.9  98.4| 0.00    197    187      3     10| 93.4  83.5 

11:clasp2 - B-SSL-alpha
Average Precision: 0.6430
 F1  Rcll  Prcn|  FAR     GT     TP     FP     FN| MODA  MODP 
77.5  67.4  91.2| 0.00    138     93      9     45| 60.9  80.2    

-------------------------
  
  
"""
out_path  = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/' \
            'tracking_wo_bnw/data/CLASP/train_gt_all/metric_bars'
if not os.path.exists(out_path):
    os.makedirs(out_path)
# data to plot
n_groups = 4
for metric in ['F1','Recall', 'Precision', 'AP', 'MODA',  'MODP']:
    if metric=='MODA':
        person_sota = (56.3,72.3,    61.7,58.4) #(9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (90.3,97.0,    86.0,87.3)
        person_ssl_alpha = (91.8,96.3, 88.4, 93.4)
        person_sl = (92.9, 96.5, 89.4, 90.4)

        bag_sota    = (32.4,30.1,    31.9,15.9)
        bag_ssl    = (70.6,66.4,    69.5,54.3)
        bag_ssl_alpha = (75.3, 65.2, 78.2,60.9)
        bag_sl = (80.3, 81.7, 90.2, 81.2)
    if metric=='MODP':
        person_sota = (81.1,83.1,    83.5,83.2) #(9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (82.5,83.7,    83.0,82.7)
        person_ssl_alpha = (83.8,85.0, 82.4, 83.5)
        person_sl = (85.5, 86.3, 85.9, 84.6)

        bag_sota    = (81.6,78.9,    75.1,79.8)
        bag_ssl    = (79.2,77.2,    76.9,80.3)
        bag_ssl_alpha = (80.3, 77.3, 78.3, 80.2)
        bag_sl = (84.4, 82.2, 85.4, 84.5)

    if metric=='F1':
        person_sota = (76.9,84.4,    79.8,76.3) #(9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (95.1,98.5,    92.8,93.6)
        person_ssl_alpha = (95.9,98.1, 94.2, 96.6)
        person_sl = (96.4, 98.2, 94.7, 95.0)

        bag_sota    = (50.7,51.7,    56.3,34.8)
        bag_ssl    = (84.1,81.6,    82.6,72.2)
        bag_ssl_alpha = (86.7, 81.6, 88.8, 77.5)
        bag_sl = (90.1, 90.7, 95.1, 90.6)

    if metric=='Precision':
        person_sota = (81.8,96.5,    84.3,88.6) #(9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (96.1,99.2,    95.0,93.9)
        person_ssl_alpha = (95.7,98.9, 93.9, 98.4)
        person_sl = (97.2, 98.9, 94.8, 97.8)

        bag_sota    = (93.7,83.8,    78.5,77.5)
        bag_ssl    = (91.7,89.9,    96.0,92.1)
        bag_ssl_alpha = (93.8, 86.6, 91.1, 91.2)
        bag_sl = (90.3, 92.5, 94.6, 90.6)
    if metric=='Recall':
        person_sota = (72.5,75.0,    75.8,67.0) #(9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (94.1,97.8,    90.8,93.4)
        person_ssl_alpha = (96.2,97.4, 94.5, 94.9)
        person_sl = (95.7, 97.6, 94.5, 92.4)

        bag_sota    = (34.8,37.4,    43.9,22.5)
        bag_ssl    = (77.6,74.8,    72.5,59.4)
        bag_ssl_alpha = (80.6, 77.1, 86.6, 67.4)
        bag_sl = (90.0, 89.0, 95.6, 90.6)

    if metric=='AP':
        person_sota = (68.1,75.2,    73.1,64.5) #(9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (92.1,97.3,    89.7,92.2)
        person_ssl_alpha = (94.2,94.6, 91.8, 92.6)
        person_sl = (94.8, 97.4, 92.2, 90.2)

        bag_sota    = (33.0,34.8,    40.6,17.4)
        bag_ssl    = (77.5,72.4,    69.9,57.6)
        bag_ssl_alpha = (80.0,73.5, 84.5, 64.3)
        bag_sl = (87.3, 87.4, 94.7, 88.2)

    # create plot
    fig, ax = plt.subplots()
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    index = np.arange(n_groups) # [0, 1, 2, 3]
    bar_width = 0.11
    opacity_sota = 0.5
    opacity_ours = 0.5
    plt.rc('font', size=18)
    rects1 = plt.bar(index, person_sota, bar_width,
                    alpha=opacity_sota,
                    color='aqua',
                    label='person_sota')

    rects2 = plt.bar(index + bar_width*(2-1), person_ssl, bar_width,
                     alpha=opacity_ours,
                     color='green',
                     label='person_ssl')


    rects3 = plt.bar(index + bar_width*(3-1), person_ssl_alpha, bar_width,
                     alpha=opacity_ours + 0.3,
                     color='green',
                     label='person_ssl_alpha')

    rects4 = plt.bar(index + bar_width*(4-1) , person_sl, bar_width,
                     alpha=opacity_ours,
                     color='blue',
                     label='person_sl')

    rects5 = plt.bar(index + bar_width*(5-1), bag_sota, bar_width,
                    alpha=opacity_sota,
                    color='orange',
                    label='bag_sota')
    rects6 = plt.bar(index + bar_width*(6-1), bag_ssl, bar_width,
                     alpha=opacity_ours,
                     color='magenta',
                     label='bag_ssl')

    rects7 = plt.bar(index + bar_width*(7-1), bag_ssl_alpha, bar_width,
                     alpha=opacity_ours + 0.3,
                     color='magenta',
                     label='bag_ssl_alpha')

    rects8 = plt.bar(index + bar_width*(8-1), bag_sl, bar_width,
                     alpha=opacity_ours,
                     color='black',
                     label='bag_sl')

    #plt.xlabel('Person')
    uparrow = u'\u2191'
    plt.xlabel('{}{}'.format(uparrow, metric),fontsize=20)
    #plt.title('Scores by person')
    plt.ylim([0, 100])
    plt.xticks(index+bar_width+bar_width+0.16, ('9:CL1','11:CL1','9:CL2', '11:CL2'),fontsize=18)#'9:C', '11:C','9:D', '11:D','9:E', '11:E'
    plt.yticks(fontsize=18)
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    #plt.xticks(1 + bar_width, ('\nMRCNN+MHT'))
    #plt.xticks(5 + bar_width, ('\nOurs'))
    plt.legend(('P-baseline', 'P-SSL-wo-'+r'$\alpha$', 'P-SSL', 'P-SL',
                'B-baseline', 'B-SSL-wo-'+r'$\alpha$', 'B-SSL', 'B-SL'),
               loc='lower left',ncol=2, prop={'size': 12})

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'Det_{}_SSL.svg'.format(metric)))
    plt.show()
    plt.waitforbuttonpress
    plt.close()

