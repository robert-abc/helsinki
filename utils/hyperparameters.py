def get_hyperparam(deblur_level):
    l=[
        {'radius':0,
         'width'=512,
         'num_iter'=1500,
         'iter_lr'=[200,400,600],
         'iter_dl'=[1000,1100,1200,1300],
         'iter_mean'=1400,
         'dl_param'=[1e-2,1e-2,5e-3,5e-3]
        }
    ]
