# BP functions

# prereq
import torch

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

def BPTokens_storage(bpsize, bpPortion,l1_act, l2_act, shape_act, color_act, location_act, shape_coeff, color_coeff, location_coeff, l1_coeff,l2_coeff, bs_testing, normalize_fact, std=1):
    notLink_all = list()  # will be used to accumulate the specific token linkages
    BP_in_all = list()  # will be used to accumulate the bp activations for each item
    tokenBindings = list()
    bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
    bp_in_color_dim = color_act.shape[1]
    bp_in_location_dim = location_act.shape[1]
    bp_in_L1_dim = l1_act.shape[1]
    bp_in_L2_dim = l2_act.shape[1]
    #std = 1
    shape_fw = torch.randn(bp_in_shape_dim,
                            bpsize).cuda() *std  # make the randomized fixed weights to the binding pool
    color_fw = torch.randn(bp_in_color_dim, bpsize).cuda() *std
    location_fw = torch.randn(bp_in_location_dim, bpsize).cuda()*std
    L1_fw = torch.randn(bp_in_L1_dim, bpsize).cuda() *std
    L2_fw = torch.randn(bp_in_L2_dim, bpsize).cuda() *std

    # ENCODING!  Store each item in the binding pool
    for items in range(bs_testing):  # the number of images
        tkLink_tot = torch.randperm(bpsize)  # for each token figure out which connections will be set to 0
        notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

        BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coeff + torch.mm(
        color_act[items, :].view(1, -1), color_fw) * color_coeff + torch.mm(
        location_act[items, :].view(1, -1), location_fw) * location_coeff + torch.mm(
        l1_act[items, :].view(1, -1), L1_fw) * l1_coeff + torch.mm(l2_act[items, :].view(1, -1), L2_fw) * l2_coeff

        BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
        BP_in_all.append(BP_in_eachimg)  # appending and stacking images
        notLink_all.append(notLink)
    # now sum all of the BPs together to form one consolidated BP activation set.
    BP_in_items = torch.stack(BP_in_all)
    BP_in_items = torch.squeeze(BP_in_items, 1)
    BP_in_items = torch.sum(BP_in_items, 0).view(1, -1)  # Add them up
    tokenBindings.append(torch.stack(notLink_all))  # this is the set of 0'd connections for each of the tokens
    tokenBindings.append(shape_fw)
    tokenBindings.append(color_fw)
    tokenBindings.append(location_fw)
    tokenBindings.append(L1_fw)
    tokenBindings.append(L2_fw)

    return BP_in_items, tokenBindings



def BPTokens_retrieveByToken( bpsize, bpPortion, BP_in_items,tokenBindings, l1_act, l2_act, shape_act, color_act, location_act,bs_testing,normalize_fact):
# NOW REMEMBER THE STORED ITEMS
    #notLink_all = list()  # will be used to accumulate the specific token linkages
    BP_in_all = list()  # will be used to accumulate the bp activations for each item
    notLink_all = tokenBindings[0]
    shape_fw = tokenBindings[1]
    color_fw = tokenBindings[2]
    location_fw = tokenBindings[3]
    L1_fw = tokenBindings[4]
    L2_fw = tokenBindings[5]
   
    tokenBindings.append(shape_fw)
    tokenBindings.append(color_fw)
    tokenBindings.append(location_fw)
    tokenBindings.append(L1_fw)
    tokenBindings.append(L2_fw)

    bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
    bp_in_color_dim = color_act.shape[1]
    bp_in_location_dim = location_act.shape[1]
    bp_in_L1_dim = l1_act.shape[1]
    bp_in_L2_dim = l2_act.shape[1]

    shape_out_all = torch.zeros(bs_testing,
                                bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
    color_out_all = torch.zeros(bs_testing,
                                bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
    location_out_all = torch.zeros(bs_testing,
                                bp_in_location_dim).cuda()  # will be used to accumulate the reconstructed location
    L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
    L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()
    BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix to the number of items to easier retrieve
    for items in range(bs_testing):  # for each item to be retrieved
        BP_in_items[items, notLink_all[items, :]] = 0  # set the BPs to zero for this token retrieval
        L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
        L1_out_all[items,:] = L1_out_eachimg / bpPortion  # put the reconstructions into a big tensor and then normalize by the effective # of BP nodes

        L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
        L2_out_all[items, :] = L2_out_eachimg / bpPortion  #

        shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), shape_fw.t()).cuda()  # do the actual reconstruction
        color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
        location_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), location_fw.t()).cuda()
        shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
        color_out_all[items, :] = color_out_eachimg / bpPortion
        location_out_all[items, :] = location_out_eachimg / bpPortion

    return shape_out_all, color_out_all, location_out_all, L2_out_all, L1_out_all