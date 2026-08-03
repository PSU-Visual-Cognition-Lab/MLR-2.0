import simulation_src.figures as figures
import torch
import os
# panels:

@torch.no_grad()
def interference(vae, folder_path, load_data=False):
    folder_path = folder_path + "interference/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_feature_swap(vae, folder_path, load_data)
    figures.fig_efficient_rep(vae, folder_path) #WORKING
    pass

@torch.no_grad()
def individuated(vae, folder_path, load_data=False):
    folder_path = folder_path + "individuated/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_repeat_recon(vae, folder_path)
    figures.fig_non_repeat_recon(vae, folder_path)
    figures.fig_non_color_repeat_recon(vae, folder_path)
    pass

@torch.no_grad()
def novel(vae, folder_path, load_data=False):
    folder_path = folder_path + "novel/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_novel_representations(vae, folder_path)
    pass

@torch.no_grad()
def addressability(vae, color_classifier, folder_path, load_data=False):
    folder_path = folder_path + "addressability/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_binding_addressability(vae, color_classifier, folder_path)

@torch.no_grad()
def generative(vae, shape_label, s_classes, color_label, c_classes, folder_path, load_data=False):
    folder_path = folder_path + "generative/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # TODO: implement fig_generative_noise()
    figures.fig_generative_noise(vae, shape_label, s_classes, color_label, c_classes, folder_path, load_data)

@torch.no_grad()
def synthesis(vae, shape_label, s_classes, shape_classifier, folder_path, load_data=False):
    folder_path = folder_path + "synthesis/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # head phones 2 o's plus U, lolipop O + i, ice cream cone V + 1-3 o's
    #figures.fig_visual_synthesis_umbrella(vae, shape_label, s_classes, shape_classifier, folder_path + "umbrella/", load_data)
    figures.fig_visual_synthesis_clock(vae, shape_label, s_classes, shape_classifier, folder_path + "clock/", load_data)
    #figures.fig_visual_synthesis_boat(vae, shape_label, s_classes, shape_classifier, folder_path + "boat/", load_data)

@torch.no_grad()
def scene(vae, object_label, color_label, object_classifier, color_classifier, folder_path, load_data=False):
    folder_path = folder_path + "scene/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_obj_scene_recon(vae, object_label, color_label, object_classifier, color_classifier, folder_path, load_data)

@torch.no_grad()
def compositional(vae, folder_path, load_data=False):
    folder_path = folder_path + "compositional/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_retinal_mod(vae, folder_path)

@torch.no_grad()
def flexibility(vae, folder_path, load_data=False):
    folder_path = folder_path + "flexibility/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_encoding_flexibility(vae, folder_path)

@torch.no_grad()
def holistic(vae, folder_path, load_data=False):
    folder_path = folder_path + "holistic/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_retinal_mod(vae, folder_path)
    #pass

@torch.no_grad()
def poster(vae, folder_path, load_data=False):
    folder_path = folder_path + "poster/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    figures.fig_simultaneous_vs_sequential(vae, folder_path, load_data)
    #figures.fig_efficient_rep(vae, folder_path) #WORKING

@torch.no_grad()
def basic(vae, folder_path, load_data=False):
    folder_path = folder_path + "basic/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    figures.functionality_test(vae, 0, 0, 0, 0, folder_path)
    figures.recon_test(vae, folder_path)

@torch.no_grad()
def modality(vae, shape_label, s_classes, folder_path, load_data=False):
    folder_path = folder_path + "modality/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    figures.percept_concept(vae, shape_label, s_classes, folder_path, load_data)

