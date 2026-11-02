# BP functions

# prereq
import torch
from collections import defaultdict

def BPTokens_with_labels(bp_outdim, bpPortion,storeLabels, shape_coef, color_coef, shape_act, color_act,l1_act,l2_act,oneHotShape, oneHotcolor, bs_testing, layernum, normalize_fact ):
    # Store and retrieve multiple items including labels in the binding pool
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items

    with torch.no_grad():  # <---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all = list()  # will be used to accumulate the specific token linkages
        BP_in_all = list()  # will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]
        bp_in_L2_dim = l2_act.shape[1]
        oneHotShape = oneHotShape.cuda()

        oneHotcolor = oneHotcolor.cuda()
        bp_in_Slabels_dim = oneHotShape.shape[1]  # dim =20
        bp_in_Clabels_dim= oneHotcolor.shape[1]


        shape_out_all = torch.zeros(bs_testing,bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out_all = torch.zeros(bs_testing,bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
        L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()
        shape_label_out=torch.zeros(bs_testing, bp_in_Slabels_dim).cuda()
        color_label_out = torch.zeros(bs_testing, bp_in_Clabels_dim).cuda()

        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  # make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()
        L2_fw = torch.randn(bp_in_L2_dim, bp_outdim).cuda()
        shape_label_fw=torch.randn(bp_in_Slabels_dim, bp_outdim).cuda()
        color_label_fw = torch.randn(bp_in_Clabels_dim, bp_outdim).cuda()

        # ENCODING!  Store each item in the binding pool
        for items in range(bs_testing):  # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

            if layernum == 1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            elif layernum==2:
                BP_in_eachimg = torch.mm(l2_act[items, :].view(1, -1), L2_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coef + torch.mm(color_act[items, :].view(1, -1), color_fw) * color_coef  # binding pool inputs (forward activations)
                BP_in_Slabels_eachimg=torch.mm(oneHotShape [items, :].view(1, -1), shape_label_fw)
                BP_in_Clabels_eachimg = torch.mm(oneHotcolor[items, :].view(1, -1), color_label_fw)


            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_Slabels_eachimg[:, notLink] = 0
            BP_in_Clabels_eachimg[:, notLink] = 0
            if storeLabels==1:
                BP_in_all.append(
                    BP_in_eachimg + BP_in_Slabels_eachimg + BP_in_Clabels_eachimg)  # appending and stacking images
                notLink_all.append(notLink)

            else:
                BP_in_all.append(BP_in_eachimg )  # appending and stacking images
                notLink_all.append(notLink)



        # now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items, 1)
        BP_in_items = torch.sum(BP_in_items, 0).view(1, -1)  # divide by the token percent, as a normalizing factor

        BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix to the number of items to easier retrieve
        notLink_all = torch.stack(notLink_all)  # this is the set of 0'd connections for each of the tokens

        # NOW REMEMBER
        for items in range(bs_testing):  # for each item to be retrieved
            BP_in_items[items, notLink_all[items, :]] = 0  # set the BPs to zero for this token retrieval
            if layernum == 1:
                L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
                L1_out_all[items,:] = (L1_out_eachimg / bpPortion ) * normalize_fact # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
            if layernum==2:

                L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
                L2_out_all[items, :] = L2_out_eachimg / bpPortion  #
            else:
                shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),shape_fw.t()).cuda()  # do the actual reconstruction
                color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
                shapelabel_out_each=torch.mm(BP_in_items[items, :].view(1, -1),shape_label_fw.t()).cuda()
                colorlabel_out_each = torch.mm(BP_in_items[items, :].view(1, -1), color_label_fw.t()).cuda()

                shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
                color_out_all[items, :] = color_out_eachimg / bpPortion
                shape_label_out[items,:]=shapelabel_out_each/bpPortion
                color_label_out[items,:]=colorlabel_out_each/bpPortion

    return shape_out_all, color_out_all, L2_out_all, L1_out_all,shape_label_out,color_label_out


def BPTokens_binding_all(bp_outdim,  bpPortion, shape_coef,color_coef,shape_act,color_act,l1_act,bs_testing,layernum, shape_act_grey, color_act_grey):
    #Store multiple items in the binding pool, then try to retrieve the token of item #1 using its shape as a cue
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items
    #layernum= either 1 (reconstructions from l1) or 0 (recons from the bottleneck
    with torch.no_grad(): #<---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all=list()  #will be used to accumulate the specific token linkages
        BP_in_all=list()    #will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottlenecks
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]  # neurons in the Bottleneck
        tokenactivation = torch.zeros(bs_testing)  # used for finding max token
        shape_out = torch.zeros(bs_testing,
                                    bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out= torch.zeros(bs_testing,
                                    bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        l1_out= torch.zeros(bs_testing, bp_in_L1_dim).cuda()


        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  #make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()

        #ENCODING!  Store each item in the binding pool
        for items in range (bs_testing):   # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  #list of 0'd BPs for this token
            if layernum==1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw)+torch.mm(color_act[items, :].view(1, -1), color_fw) # binding pool inputs (forward activations)

            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_all.append(BP_in_eachimg)  # appending and stacking images
            notLink_all.append(notLink)

        #now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items,1)
        BP_in_items = torch.sum(BP_in_items,0).view(1,-1)   #divide by the token percent, as a normalizing factor

        notLink_all=torch.stack(notLink_all)   # this is the set of 0'd connections for each of the tokens

        retrieve_item = 0
        if layernum==1:
            BP_reactivate = torch.mm(l1_act[retrieve_item, :].view(1, -1), L1_fw)
        else:
            BP_reactivate = torch.mm(shape_act_grey[retrieve_item, :].view(1, -1),shape_fw)  # binding pool retreival

        # Multiply the cued version of the BP activity by the stored representations
        BP_reactivate = BP_reactivate  * BP_in_items

        for tokens in range(bs_testing):  # for each token
            BP_reactivate_tok = BP_reactivate.clone()
            BP_reactivate_tok[0,notLink_all[tokens, :]] = 0  # set the BPs to zero for this token retrieval
            # for this demonstration we're assuming that all BP-> token weights are equal to one, so we can just sum the
            # remaining binding pool neurons to get the token activation
            tokenactivation[tokens] = BP_reactivate_tok.sum()

        max, maxtoken =torch.max(tokenactivation,0)   #which token has the most activation

        BP_in_items[0, notLink_all[maxtoken, :]] = 0  #now reconstruct color from that one token
        if layernum==1:

            l1_out = torch.mm(BP_in_items.view(1, -1), L1_fw.t()).cuda() / bpPortion  # do the actual reconstruction
        else:

            shape_out = torch.mm(BP_in_items.view(1, -1), shape_fw.t()).cuda() / bpPortion  # do the actual reconstruction of the BP
            color_out = torch.mm(BP_in_items.view(1, -1), color_fw.t()).cuda() / bpPortion

    return tokenactivation, maxtoken, shape_out,color_out, l1_out

def get_act_names(token_act_bitmask, act_name_map):
    act_names = []
    for i in range(len(token_act_bitmask)):
        if token_act_bitmask[i]:
            act_names += [act_name_map[i]]
    return act_names


# input a bitmask of which latents to encode/decode per token
# bitmasks: {token:mask} eg: {0:0110101...}

# activation dict format: {act_name: [act, coeff]}

# input_dict = {act_name: [act, coeff]}, l1 (skip), shape, color, spatial
# bpsize: size of binding pool: int, bpPortion: number of units per token: int, bs_testing: set size, number of objects(tokens):int
# normalize_fact: was used to normalize activations, currently deprecated, std: variance of the fixed random weights
def BPTokens_storage_bitmask(bpsize, bpPortion, activation_dict, bs_testing, normalize_fact, std=1):
    notLink_all = list()  # will be used to accumulate the specific token linkages
    BP_in_all = list()  # will be used to accumulate the bp activations for each item
    tokenBindings = {}
    act_count = 7

    if 'act_bitmask' in activation_dict:
        act_bitmask = activation_dict['act_bitmask']
        act_name_map = activation_dict['act_name_map']

    else: # maintains original behavior for figures relying on the old implementation
        act_bitmask = {x: [1] * act_count for x in range(bs_testing)}
        act_name_map = ['shape', 'color', 'location', 'scale', 'l1', 'l2', 'object']

    #activations = defaultdict(lambda: [torch.zeros([bs_testing,1]).cuda(), 0], activation_dict) # error handling, default 0 coeffs
    # activation multi-hot tensor to track which reps are being stored
    activations = {}
    for act_name in act_name_map:
        act, coeff = activation_dict[act_name]
        bp_in_dim = act.shape[1]
        act_fw = torch.randn(bp_in_dim, bpsize).cuda() *std
        activations[act_name] = [act, coeff, bp_in_dim, act_fw]

    per_token_contents = defaultdict(list)
    token_act_indices = defaultdict(int)
    for token in range(bs_testing):
        obj_activations_bitmask = act_bitmask[token]
        act_names = get_act_names(obj_activations_bitmask, act_name_map)
        for act_name in act_names:
            act, coeff, bp_in_dim, act_fw = activations[act_name]
            per_token_contents[token] += [[act[token_act_indices[act_name]].view(1, -1), coeff, act_fw]]
            token_act_indices[act_name] += 1

    # ENCODING!  Store each item in the binding pool
    for items in range(bs_testing):  # the number onf images
        tkLink_tot = torch.randperm(bpsize)  # for each token figure out which connections will be set to 0
        notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token
        BP_in_eachimg = sum(torch.mm(act, act_fw) * coeff for (act, coeff, act_fw) in per_token_contents[items])
        BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
        BP_in_all.append(BP_in_eachimg)  # appending and stacking images
        notLink_all.append(notLink)
    # now sum all of the BPs together to form one consolidated BP activation set.
    BP_activation = torch.stack(BP_in_all)
    BP_activation = torch.squeeze(BP_activation, 1)
    BP_activation = torch.sum(BP_activation, 0).view(1, -1)  # Add them up
    # 
    tokenBindings['notLink_all'] = torch.stack(notLink_all)  # this is the set of 0'd connections for each of the tokens
    for act_name in activations:
        tokenBindings[act_name] = (activations[act_name][3]) # fw weights

    return BP_activation, tokenBindings

def BPTokens_retrieveByToken_bitmask(bpsize, bpPortion, BP_in_items, tokenBindings, activation_dict, bs_testing, normalize_fact):
    notLink_all = tokenBindings['notLink_all']

    if 'act_bitmask' in activation_dict:
        act_bitmask = activation_dict['act_bitmask']
        act_name_map = activation_dict['act_name_map']
    else:  # maintains original behavior for figures relying on the old implementation
        act_count = 7
        act_bitmask = {x: [1] * act_count for x in range(bs_testing)}
        act_name_map = ['shape', 'color', 'location', 'scale', 'l1', 'l2', 'object']

    # forward weights per act_name, saved off during encoding
    fw_by_name = {act_name: tokenBindings[act_name] for act_name in act_name_map if act_name in tokenBindings}

    # figure out, per token, which act_names are "on" -- same bitmask logic as storage
    token_act_names = {}
    act_instance_count = defaultdict(int)
    for token in range(bs_testing):
        obj_activations_bitmask = act_bitmask[token]
        act_names = get_act_names(obj_activations_bitmask, act_name_map)
        token_act_names[token] = act_names
        for act_name in act_names:
            act_instance_count[act_name] += 1

    # destination tensors -- one row per instance of that act_name across all tokens,
    # matching how storage incremented token_act_indices[act_name] per occurrence
    out_all = {}
    for act_name, fw in fw_by_name.items():
        bp_in_dim = fw.shape[0]
        out_all[act_name] = torch.zeros(act_instance_count[act_name], bp_in_dim).cuda()

    # Decoding! Retrieve each item from the binding pool
    BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat so each token gets its own row to retrieve from
    token_act_indices = defaultdict(int)
    for token in range(bs_testing):
        BP_in_items[token, notLink_all[token, :]] = 0  # zero out the unconnected BPs for this token's retrieval

        for act_name in token_act_names[token]:
            fw = fw_by_name[act_name]
            out_eachimg = torch.mm(BP_in_items[token, :].view(1, -1), fw.t()).cuda()

            idx = token_act_indices[act_name]
            if act_name == 'l1':  # l1 keeps its special normalize_fact scaling from the original
                out_all[act_name][idx, :] = (out_eachimg / bpPortion) * normalize_fact
            else:
                out_all[act_name][idx, :] = out_eachimg / bpPortion
            token_act_indices[act_name] += 1

    return out_all

# OLD implementation
def BPTokens_storage(bpsize, bpPortion, activation_dict, bs_testing, normalize_fact, std=1):
    notLink_all = list()  # will be used to accumulate the specific token linkages
    BP_in_all = list()  # will be used to accumulate the bp activations for each item
    tokenBindings = list()

    # activation dict format: {act_name: [act, coeff]}

    activations = defaultdict(lambda: [torch.zeros([bs_testing,1]).cuda(), 0], activation_dict) # error handling, default 0 coeffs
    # activation multi-hot tensor to track which reps are being stored
    shape_act, shape_coeff = activations['shape']
    color_act, color_coeff = activations['color']
    location_act, location_coeff = activations['location']
    scale_act, scale_coeff = activations['scale']
    l1_act, l1_coeff = activations['l1']
    l2_act, l2_coeff = activations['l2']
    object_act, object_coeff = activations['object']
    #print(l1_act.size())

    bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
    bp_in_color_dim = color_act.shape[1]
    bp_in_location_dim = location_act.shape[1]
    bp_in_L1_dim = l1_act.shape[1]
    bp_in_L2_dim = l2_act.shape[1]
    bp_in_scale_dim = scale_act.shape[1]
    bp_in_object_dim = object_act.shape[1]
    #std = 1
    shape_fw = torch.randn(bp_in_shape_dim,
                            bpsize).cuda() *std  # make the randomized fixed weights to the binding pool
    color_fw = torch.randn(bp_in_color_dim, bpsize).cuda() *std
    location_fw = torch.randn(bp_in_location_dim, bpsize).cuda()*std
    L1_fw = torch.randn(bp_in_L1_dim, bpsize).cuda() *std
    L2_fw = torch.randn(bp_in_L2_dim, bpsize).cuda() *std
    scale_fw = torch.randn(bp_in_scale_dim, bpsize).cuda() *std
    object_fw = torch.randn(bp_in_object_dim, bpsize).cuda() *std

    # ENCODING!  Store each item in the binding pool
    for items in range(bs_testing):  # the number of images
        tkLink_tot = torch.randperm(bpsize)  # for each token figure out which connections will be set to 0
        notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

        BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coeff + torch.mm(
            color_act[items, :].view(1, -1), color_fw) * color_coeff + torch.mm(
            location_act[items, :].view(1, -1), location_fw) * location_coeff + torch.mm(
            l1_act[items, :].view(1, -1), L1_fw) * l1_coeff + torch.mm(l2_act[items, :].view(1, -1), L2_fw) * l2_coeff + torch.mm(scale_act[items, :].view(1, -1), scale_fw) * scale_coeff + torch.mm(
            object_act[items, :].view(1, -1), object_fw) * object_coeff

        BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
        BP_in_all.append(BP_in_eachimg)  # appending and stacking images
        notLink_all.append(notLink)
    # now sum all of the BPs together to form one consolidated BP activation set.
    BP_activation = torch.stack(BP_in_all)
    BP_activation = torch.squeeze(BP_activation, 1)
    BP_activation = torch.sum(BP_activation, 0).view(1, -1)  # Add them up
    tokenBindings.append(torch.stack(notLink_all))  # this is the set of 0'd connections for each of the tokens
    tokenBindings.append(shape_fw)
    tokenBindings.append(color_fw)
    tokenBindings.append(location_fw)
    tokenBindings.append(L1_fw)
    tokenBindings.append(L2_fw)
    tokenBindings.append(scale_fw)
    tokenBindings.append(object_fw)

    return BP_activation, tokenBindings



# BP_in_items -> BP_activations
# input_dict = {act_name: [act, coeff]}, l1 (skip), shape, color, spatial
# bpsize: size of binding pool: int, bpPortion: number of units per token: int, bs_testing: set size, number of objects:int
# normalize_fact: was used to normalize activations, currently deprecated, std: variance of the fixed random weights
# BP_in_items: binding pool activation from storage, tokenBindings: forward weights used to store activations in storage
def BPTokens_retrieveByToken(bpsize, bpPortion, BP_in_items, tokenBindings, activation_dict, bs_testing, normalize_fact):
# NOW REMEMBER THE STORED ITEMS
    BP_in_all = list()  # will be used to accumulate the bp activations for each item
    notLink_all = tokenBindings[0]
    shape_fw = tokenBindings[1]
    color_fw = tokenBindings[2]
    location_fw = tokenBindings[3]
    L1_fw = tokenBindings[4]
    L2_fw = tokenBindings[5]
    scale_fw = tokenBindings[6]
    object_fw = tokenBindings[7]

    activations = defaultdict(lambda: [torch.zeros([bs_testing,1]).cuda(), 0], activation_dict) # error handling
    
    shape_act, shape_coeff = activations['shape']
    color_act, color_coeff = activations['color']
    location_act, location_coeff = activations['location']
    scale_act, scale_coeff = activations['scale']
    l1_act, l1_coeff = activations['l1']
    l2_act, l2_coeff = activations['l2']
    object_act, object_coeff = activations['object']
    #print(l1_act.size())

    bp_in_shape_dim = shape_act.shape[1]  # dimensions for each latent space in the binding model
    bp_in_color_dim = color_act.shape[1]
    bp_in_location_dim = location_act.shape[1]
    bp_in_L1_dim = l1_act.shape[1]
    bp_in_L2_dim = l2_act.shape[1]
    bp_in_scale_dim = scale_act.shape[1]
    bp_in_object_dim = object_act.shape[1]

    # destination vectors for reconstructions
    shape_out_all = torch.zeros(bs_testing, bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
    color_out_all = torch.zeros(bs_testing, bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
    location_out_all = torch.zeros(bs_testing, bp_in_location_dim).cuda()  # will be used to accumulate the reconstructed location
    L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
    L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()
    scale_out_all = torch.zeros(bs_testing, bp_in_scale_dim).cuda()
    object_out_all = torch.zeros(bs_testing, bp_in_object_dim).cuda()

    # Decoding!  Retrieve each item from the binding pool
    BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix so the number of items to easier retrieve
    for items in range(bs_testing):  # for each item to be retrieved
        BP_in_items[items, notLink_all[items, :]] = 0  # set the unconnected BPs to zero for this token retrieval
        L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
        L1_out_all[items,:] = (L1_out_eachimg / bpPortion) * normalize_fact  # put the reconstructions into a big tensor and then normalize by the effective # of BP nodes

        L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
        L2_out_all[items, :] = L2_out_eachimg / bpPortion

        shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), shape_fw.t()).cuda()  # do the actual reconstruction
        color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
        location_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), location_fw.t()).cuda()
        scale_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), scale_fw.t()).cuda()
        object_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), object_fw.t()).cuda()

        shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
        color_out_all[items, :] = color_out_eachimg / bpPortion
        location_out_all[items, :] = location_out_eachimg / bpPortion
        scale_out_all[items, :] = scale_out_eachimg / bpPortion
        object_out_all[items, :] = object_out_eachimg / bpPortion

    return {'shape':shape_out_all, 'color':color_out_all, 
            'location':location_out_all, 'scale':scale_out_all, 
            'l2':L2_out_all, 'l1':L1_out_all, 'object':object_out_all}

# store both tok1,tok2, probe, take whichever token is higher

#def ()
