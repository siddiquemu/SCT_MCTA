#######################################
#
# Script for plotting tracking performance of MRCNN+MHT using Bar diagram
#
# Tracking Results from MHT: %MT = MT/GT, PT?? : should be - (MT+PT)/GT
#######################################
import numpy as np
import matplotlib.pyplot as plt
import os

out_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/' \
           'tracking_wo_bnw/data/CLASP/train_gt_all/metric_bars'
if not os.path.exists(out_path):
    os.makedirs(out_path)
# data to plot
# use average of the individual sequence results instead of the tools overall
n_groups = 4
for metric, leg_loc in zip(['IDF1', 'IDP', 'IDR', 'MT', 'IDs', 'FM', 'MOTA'],
                           ['lower left', 'lower left', 'lower left', 'lower left', 'upper left', 'upper left', 'lower left']):
    if metric == 'IDF1':
        person_sota = (88.2, 88.8, 79.4,81.7)  # (9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (91.3,93.9, 82.8,83.3)
        person_ssl_alpha = (91.8,93.7,  84.2,83.9 )
        person_sl = (90.4, 93.3, 83.1, 81.4)

        bag_sota = (76.8,75.9, 66.0,54.2)
        bag_ssl = (83.9,85.1, 75.6,71.1)

        bag_ssl_alpha = (82.2,85.8, 81.5,75.8)
        bag_sl = (86.4,90.1, 82.6,82.2)
    if metric == 'IDP':
        person_sota = (86.9,86.8, 83.8,86.2)  # (9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (91.9,93.8, 83.6,87.2 )
        person_ssl_alpha = (92.0,93.7, 84.7,85.0)
        person_sl = (90.2,93.3, 84.2,83.3)

        bag_sota = (92.6,90.4, 75.2,87.6)
        bag_ssl = (84.6,86.4, 84.2,86.7)

        bag_ssl_alpha = (81.9,85.7, 85.1,89.2)
        bag_sl = (85.4,90.6, 80.2,80.2)
    if metric == 'IDR':
        person_sota = (89.6,90.9, 75.6,77.1)  # (9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (90.6,93.9, 82.0,82.4 )
        person_ssl_alpha = (91.7,93.7, 83.7,82.8 )
        person_sl = (90.5,93.3, 82.2,79.6)

        bag_sota = (66.0,65.4, 60.3,39.8)
        bag_ssl = (83.3,84.4, 68.3,60.8)

        bag_ssl_alpha = (82.7,86.7, 77.7,65.9)
        bag_sl = (87.6,89.7, 84.7,84.5)
    if metric == 'MT':
        person_sota = (89.4,95.2, 57.3,65.5)  # (9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (90.5,98.4, 70.6,79.5 )
        person_ssl_alpha = (94.1,98.4, 73.3,79.5)
        person_sl = (95.3,96.8, 77.3,79.5)

        bag_sota = (47.8,50.0, 50.0,25.9)
        bag_ssl = (71.7,73.8, 62.2,50.6)

        bag_ssl_alpha = (71.7,78.5, 70.0,51.9 )
        bag_sl = (80.4,83.3,  92.2,92.2)
    if metric == 'IDs':
        person_sota = (20.0,19.0, 26.6,25.8)  # (9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (17.6,15.8, 22.6,22.5 )
        person_ssl_alpha = (20.0,14.2, 27.3,20.4)
        person_sl = (17.6,9.5, 28.6,23.6)

        bag_sota = (17.3,23.8, 17.7,9.0)
        bag_ssl = (32.6,33.3, 22.2,28.5)

        bag_ssl_alpha = (45.6,28.5, 21.1,20.7)
        bag_sl = (39.1,28.5, 55.5,29.8)
    if metric == 'FM':
        person_sota = (45.8, 34.9, 46.0,23.6)  # (9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (38.8,26.9, 40.0,13.9 )
        person_ssl_alpha = (37.6,26.9, 40.6,12.9)
        person_sl = (32.9,20.6, 44.6,13.9)

        bag_sota = (19.5,38.0,  66.6,5.2)
        bag_ssl = (52.1,33.3, 61.1,15.5)

        bag_ssl_alpha = (54.3,26.1, 57.7,11.6)
        bag_sl = (39.1,42.8, 22.2,16.8)
    if metric == 'MOTA':
        person_sota = (87.1,87.2, 79.2,82.3)  # (9:clasp1, 11:clasp1, 9:clasp2, 11:clasp2)
        person_ssl = (93.4,95.9, 82.5,88.7 )
        person_ssl_alpha = (94.3,97.8, 82.2,89.8,)
        person_sl = (93.0,96.7, 83.9,88.9 )

        bag_sota = (67.7,63.0,  49.2,37.7)
        bag_ssl = (79.5,79.8,  68.8,63.6)

        bag_ssl_alpha = (80.9,78.2, 76.3,65.1)
        bag_sl = (85.6,87.2, 83.8,76.2)
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)  # [0, 1, 2, 3]
    bar_width = 0.11
    opacity_sota = 0.5
    opacity_ours = 0.5
    plt.rc('font', size=18)
    rects1 = plt.bar(index, person_sota, bar_width,
                     alpha=opacity_sota,
                     color='aqua',
                     label='person_sota')

    rects2 = plt.bar(index + bar_width * (2 - 1), person_ssl, bar_width,
                     alpha=opacity_ours,
                     color='green',
                     label='person_ssl')


    rects3 = plt.bar(index + bar_width * (3 - 1), person_ssl_alpha, bar_width,
                     alpha=opacity_ours + 0.3,
                     color='green',
                     label='person_ssl_alpha')

    rects4 = plt.bar(index + bar_width * (4 - 1), person_sl, bar_width,
                     alpha=opacity_ours,
                     color = 'blue',
                     label = 'person_sl')

    rects5 = plt.bar(index + bar_width * (5 - 1), bag_sota, bar_width,
                     alpha=opacity_sota,
                     color='orange',
                     label='bag_sota')

    rects6 = plt.bar(index + bar_width * (6 - 1), bag_ssl, bar_width,
                     alpha=opacity_ours,
                     color='magenta',
                     label='bag_ssl')

    rects7 = plt.bar(index + bar_width * (7 - 1), bag_ssl_alpha, bar_width,
                     alpha=opacity_ours + 0.3,
                     color='magenta',
                     label='bag_ssl_alpha')

    rects8 = plt.bar(index + bar_width * (8 - 1), bag_sl, bar_width,

                    alpha = opacity_ours,
                    color = 'black',
                    label = 'bag_sl')

    # plt.xlabel('Person')
    if metric in ['FM', 'IDs']:
        downarrow = r'$\downarrow$'
        plt.xlabel('{}{}'.format(downarrow, metric), fontsize=20)
    else:
        uparrow = r'$\uparrow$' #u'\u2191'
        plt.xlabel('{}{}'.format(uparrow, metric), fontsize=20)



    # plt.title('Scores by person')
    plt.ylim([0, 100])
    plt.xticks(index + bar_width + bar_width + 0.16, ('9:CL1', '11:CL1', '9:CL2', '11:CL2'),
               fontsize=18)  # '9:C', '11:C','9:D', '11:D','9:E', '11:E'
    plt.yticks(fontsize=18)
    # plt.xticks(1 + bar_width, ('\nMRCNN+MHT'))
    # plt.xticks(5 + bar_width, ('\nOurs'))
    plt.legend(('P-baseline', 'P-SSL-wo-'+r'$\alpha$', 'P-SSL', 'P-SL',
                'B-baseline', 'B-SSL-wo-'+r'$\alpha$', 'B-SSL','B-SL'),
               loc=leg_loc, ncol=2, prop={'size': 12})

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'Track_{}_SSL.svg'.format(metric)))
    plt.show()
    plt.waitforbuttonpress
    plt.close()

