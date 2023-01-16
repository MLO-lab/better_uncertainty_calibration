from experiments import Experiments
from pathlib import Path

Path("plots/").mkdir(parents=True, exist_ok=True)

# we only plot ~50% of whats possible
# if you want to see more plots, uncomment the respective parts

# estimators and settings to consider
ce_types = [
    'BS', 'KDE CE', 'KS CE', 'd EM 15b TCE', 'ECE', '100b TCE', '15b CWCE',
    '100b CWCE']
settings_TS_ETS = [
    'densenet40_c10', 'densenet161_imgnet', 'resnet110_SD_c100',
    'densenet40_c100', 'lenet5_c10', 'lenet5_c100', 'resnet_wide32_c10',
    'resnet_wide32_c100', 'resnet50_birds', 'resnet110_c10', 'resnet110_c100',
    'resnet110_SD_c10', 'resnet152_imgnet', 'resnet152_SD_SVHN']
settings_DIAG = [
    'densenet40_c10', 'densenet161_imgnet', 'pnasnet5_large_imgnet',
    'densenet40_c100', 'resnet101_cars', 'resnet101pre_cars',
    'resnet_wide32_c10', 'resnet_wide32_c100', 'resnet50_nts_birds',
    'resnet110_c10', 'resnet110_c100', 'resnet50pre_cars', 'resnet152_imgnet',
    'resnet152_sd_svhn']

#############################
# ALL
#############################
parent = 'all'
exp = Experiments()
exp.load('results/results_TS.csv')
exp.load('results/results_ETS.csv')

main_font_scale = 1.25
main_size = (5, 4)
size_subs = (4.5, 3)
font_scale_subs = 1.55

c10_settings = [
    'densenet40_c10', 'resnet_wide32_c10', 'resnet110_SD_c10', 'resnet110_c10',
    'lenet5_c10']
# figure 3b
exp.lineplot_rsbias(
    save_file='rbias_c10', ce_types=ce_types, settings=c10_settings,
    size=main_size, legend=True, font_scale=main_font_scale)

# figure 3a
exp.plot_CE(
    'resnet_wide32_c100', ce_types=ce_types, save_file='all_resnet_wide32_c100_large',
    font_scale=main_font_scale, size=main_size)

exp.get_legend(
    save_file='legend', ce_types=ce_types, size=size_subs, font_scale=1.75,
    padding=7)

# Appendix
for setting in settings_TS_ETS:
    exp.plot_CE(
        setting, ce_types=ce_types, save_file='{}_{}'.format(parent, setting),
        font_scale=font_scale_subs, size=size_subs)
# Appendix
for setting in ['densenet40_c10', 'resnet_wide32_c100', 'densenet161_imgnet']:
    exp.plot_CE(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='squared_{}_{}'.format(parent, setting), use_root=False)

exp = Experiments()
exp.load('results/results_DIAG.csv')

# Appendix
for setting in ['resnet50pre_cars', 'resnet50_nts_birds', 'resnet101_cars', 'resnet101pre_cars', 'pnasnet5_large_imgnet']:
    exp.plot_CE(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='{}_{}'.format(parent, setting))
    exp.plot_CE(
        setting, ce_types=ce_types,
        save_file='squared_{}_{}'.format(parent, setting), size=size_subs,
        font_scale=font_scale_subs, use_root=False)

files = ['results/results_{}.csv',]

######################## uncomment above

# exp = Experiments()
# TS and ETS have the same settings, so ETS is not needed if we dont look at
# the calibrated logits
# for parent in ['TS', 'DIAG']:
#     for file in files:
#         exp.load(file.format(parent))

# settings_c100 = [
#     'densenet40_c100', 'resnet_wide32_c100', 'resnet110_SD_c100',
#     'resnet110_c100', 'lenet5_c100']
# exp.lineplot_rsbias(
#     save_file='rbias_c100', ce_types=ce_types, settings=settings_c100,
#     size=main_size, legend=True, font_scale=main_font_scale)
#
# settings_img = [
#     'densenet161_imgnet', 'resnet152_imgnet', 'pnasnet5_large_imgnet']
# exp.lineplot_rsbias(
#     save_file='rbias_img', ce_types=ce_types, settings=settings_img,
#     size=main_size, legend=True, font_scale=main_font_scale)

# exp.lineplot_rsbias(
#     save_file='rbias_all', ce_types=ce_types, settings=c10_settings+settings_c100+settings_img,
#     size=main_size, legend=True, font_scale=main_font_scale)

# settings_cars = [
#     'resnet50pre_cars', 'resnet101_cars', 'resnet101pre_cars']
# exp.lineplot_rsbias(
#     save_file='rbias_cars', ce_types=ce_types, settings=settings_cars,
#     size=main_size, legend=True, font_scale=main_font_scale)

#############################
# TS only
#############################
parent = 'TS'
exp = Experiments()
for file in files:
    exp.load(file.format(parent))

# Appendix
for setting in settings_TS_ETS:
    exp.plot_RC_delta(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='{}_RC_{}'.format(parent, setting))
# Figure 1a
for setting in ['densenet40_c10']:  # settings_TS_ETS:
    exp.plot_RC_delta(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='squared_{}_RC_{}'.format(parent, setting), use_root=False)

# #############################
# # ETS only
# #############################
parent = 'ETS'
exp = Experiments()
for file in files:
    exp.load(file.format(parent))

# Appendix
for setting in settings_TS_ETS:  # ['resnet_wide32_c100']:  #
    exp.plot_RC_delta(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='{}_RC_{}'.format(parent, setting))
# Figure 1b
for setting in ['resnet_wide32_c100']:  # settings_TS_ETS:
    exp.plot_RC_delta(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='squared_{}_RC_{}'.format(parent, setting), use_root=False)

# #############################
# # DIAG only
# #############################
parent = 'DIAG'
exp = Experiments()
for file in files:
    exp.load(file.format(parent))

# Appendix
for setting in settings_DIAG:  # ['densenet161_imgnet']:  #
    exp.plot_RC_delta(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='{}_RC_{}'.format(parent, setting))
# Figure 1c
for setting in ['densenet161_imgnet']:  # settings_DIAG:
    exp.plot_RC_delta(
        setting, ce_types=ce_types, font_scale=font_scale_subs, size=size_subs,
        save_file='squared_{}_RC_{}'.format(parent, setting), use_root=False)
