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

out_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/' \
           'tracking_wo_bnw/data/CLASP/train_gt_all/metric_bars'
if not os.path.exists(out_path):
    os.makedirs(out_path)
# data to plot
n_groups =5
dataset = 'clasp2'
for metric in ['AP', 'MODA']:
    if dataset=='clasp1':
        if metric == 'MODA':
            p_sl = 89.4
            p_ssl = 89.5

            p_st = (88.3, 93.8, 93.1, 93.8, 95.8)  # (1%, 5%, 10%, 50%, 100%)
            p_semi_sl = (89.2,89.2,  93.8,95.6)


            b_sl = 87.7
            b_ssl = 73.1

            b_st = (69.8, 80.8, 85.3, 88.8, 88.6)  # (1%, 5%, 10%, 50%, 100%)
            b_semi_sl = (68.3,82.5,  90.7,91.6)


        if metric == 'AP':
            p_st = (94.0, 94.4, 94.3, 94.9, 95.1)  # (1%, 5%, 10%, 50%, 100%)
            p_semi_sl = (94.5,94.5,  94.7,95.0 )
            p_baseline = (40.0, 40.0, 40.0, 40.0, 40.0)

            b_st = (75.3, 84.3, 87.6, 89.1, 89.6)  # (1%, 5%, 10%, 50%, 100%)
            b_semi_sl = (70.3,87.3, 92.3,94.6)

    else:
        if metric == 'MODA':
            p_sl = 89.4
            p_ssl = 89.5

            p_st = (81.2, 86.3, 87.7, 86.7, 89.9)  # (1%, 5%, 10%, 50%, 100%)
            p_semi_sl = (86.5, 89.6, 88.0,87.5, 89.8)


            b_sl = 87.7
            b_ssl = 73.1

            b_st = (42.2, 80.8, 85.5, 90.9, 93.7)  # (1%, 5%, 10%, 50%, 100%)
            b_semi_sl = (86.1, 87.3, 88.3, 88.5, 92.7)


        if metric == 'AP':

            p_sl = 92.3
            p_ssl = 92.0

            p_st = (84.6, 87.9, 91.6, 89.6, 92.4)  # (1%, 5%, 10%, 50%, 100%)
            p_semi_sl = (89.7, 92.2, 89.8, 92.0, 94.2)


            b_sl = 92.0
            b_ssl = 79.3

            b_st = (43.1, 84.6, 89.4, 94.7, 94.6)  # (1%, 5%, 10%, 50%, 100%)
            b_semi_sl = (91.7, 91.7, 91.8, 94.3, 94.4)


    # create plot
    fig, ax = plt.subplots()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    index = np.arange(n_groups)  # [0, 1, 2, 3]
    bar_width = 0.2
    opacity_sota = 0.5
    opacity_ours = 0.5
    plt.rc('font', size=18)

        #add horizontal line for SSL and SL
    #person
    #SL
    n_step = 23.5
    plt.hlines(y=p_sl, xmin=0, xmax=n_step*0.2, linestyles='dotted', linewidth=2,  color='blue')
    #SSL
    plt.hlines(y=p_ssl, xmin=0, xmax=n_step*0.2, linestyles='dashed', linewidth=2, color='green')

    #bag
    plt.hlines(y=b_sl, xmin=0, xmax=n_step*0.2, linestyles='dotted', linewidth=2, color='black')
    plt.hlines(y=b_ssl, xmin=0, xmax=n_step*0.2, linestyles='dashed', linewidth=2, color='magenta')

    #add bars
    rects1 = plt.bar(index, p_st, bar_width,
                     alpha=opacity_sota,
                     color='aqua',
                     label='p_st')

    rects2 = plt.bar(index + bar_width * (2 - 1),p_semi_sl, bar_width,
                     alpha=opacity_ours,
                     color='green',
                     label='p_semi_sl')

    rects3 = plt.bar(index + bar_width * (3 - 1), b_st, bar_width,
                     alpha=opacity_ours,
                     color='blue',
                     label='b_st')

    rects4 = plt.bar(index + bar_width * (4 - 1), b_semi_sl, bar_width,
                     alpha=opacity_ours,
                     color='magenta',
                     label='b_semi_sl')


    uparrow = u'\u2191'
    plt.xlabel('{}{}'.format(uparrow, metric), fontsize=20)
    # plt.title('Scores by person')
    plt.ylim([0, 100])
    plt.xticks(index + bar_width + 0.1, ('1%', '5%', '10%', '50%', '100%'),
               fontsize=18)  # '9:C', '11:C','9:D', '11:D','9:E', '11:E'
    plt.yticks(fontsize=18)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    # plt.xticks(1 + bar_width, ('\nMRCNN+MHT'))
    # plt.xticks(5 + bar_width, ('\nOurs'))
    # plt.legend(('P-SoftTeacher', 'P-Semi-SL',
    #             'B-SoftTeacher', 'B-Semi-SL' ),
    #            loc='lower left', ncol=2, prop={'size': 12})
    plt.legend(('P-SL', 'P-SSL','B-SL', 'B-SSL',
                'P-ST', 'P-Semi-SL','B-ST', 'B-Semi-SL' ),
               loc='lower right', ncol=2, prop={'size': 15})

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'SoftTeacher_{metric}_SSL_{dataset}.png'))
    plt.show()
    plt.waitforbuttonpress
    plt.close()
    print(f'figure saved in {out_path}')

