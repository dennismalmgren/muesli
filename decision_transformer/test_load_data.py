from torchrl.data import GenDGRLExperienceReplay


if __name__=="main":
    gen_dgrl = GenDGRLExperienceReplay()
    gen_dgrl.load_data()
    print "Data loaded successfully.
