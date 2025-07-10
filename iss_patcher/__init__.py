"""Approximate missing features from higher dimensionality data neighbours"""
__version__ = "0.1.0"

import scipy.spatial
import scipy.sparse
import anndata
import pandas as pd
import scanpy as sc
import annoy
import pynndescent
import numpy as np

def prepare_scaled(adata, 
                   min_genes=3,
                   copy=False
                  ):
    """
    Log-normalise and z-score raw-count ``adata``, filtering to cells 
    with ``min_genes`` genes.
    """
    if copy:
        adata = adata.copy()
        
    sc.pp.filter_cells(adata, min_genes=min_genes)
    #normalise to median
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    
    if copy:
        return adata

def split_and_normalise_objects(iss, gex, 
                                min_genes=3,
                                inplace=False
                               ):
    """
    Identify shared features between ``iss`` and ``gex``, split GEX 
    into two sub-objects - ``gex`` with features shared with ``iss`` 
    and ``gex_only`` that carries GEX-unique features. Filter ``iss`` 
    and ``gex`` to cells with at least ``min_genes`` genes, 
    log-normalise and z-score them, subset ``gex_only`` to match the 
    ``gex`` cell space, return all three objects.
    
    All arguments as in ``ip.patch()``.
    """
    # Only copy if we're not modifying in place
    if not inplace:
        iss = iss.copy()
        gex = gex.copy()
    
    # Get shared gene names to avoid repeated checks
    shared_genes = [gene for gene in iss.var_names if gene in gex.var_names]
    
    # subset the ISS to genes that appear in the GEX
    iss = iss[:, shared_genes]
    
    # separate GEX into shared gene space and not shared gene space
    non_shared_genes = [gene for gene in gex.var_names if gene not in iss.var_names]
    gex_only = gex[:, non_shared_genes]
    gex = gex[:, iss.var_names]
    
    # turn both objects for KNNing into a log-normalised, z-scored form
    prepare_scaled(iss, min_genes=min_genes)
    prepare_scaled(gex, min_genes=min_genes)
    
    # this might remove some cells from the GEX, mirror in the gex_only
    gex_only = gex_only[gex.obs_names]
    return iss, gex, gex_only

def get_knn_indices(issX, gexX, 
                    computation="annoy", 
                    neighbours=15,
                    memory_efficient=True
                   ):
    """
    Identify the neighbours of each ``issX`` observation in ``gexX``. 
    Return a ``scipy.spatial.cKDTree()``-style formatted output with 
    KNN indices for each low dimensional cell in the second element 
    of the tuple.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    issX : ``np.array``
        Low dimensionality processed expression data.
    gexX : ``np.array``
        High dimensionality processed expression data, with features 
        subset to match the low dimensionality data.
    memory_efficient : ``bool``
        If True, use more memory-efficient approaches
    """
    if computation == "annoy":
        #build the GEX index
        ckd = annoy.AnnoyIndex(gexX.shape[1], metric="euclidean")
        for i in np.arange(gexX.shape[0]):
            ckd.add_item(i, gexX[i,:])
        ckd.build(10)
        
        # Memory-efficient implementation: process in batches
        if memory_efficient and issX.shape[0] > 10000:
            batch_size = 5000
            ckdo_ind = np.zeros((issX.shape[0], neighbours), dtype=np.int32)
            ckdo_dist = np.zeros((issX.shape[0], neighbours), dtype=np.float32)
            
            for i in range(0, issX.shape[0], batch_size):
                end_idx = min(i + batch_size, issX.shape[0])
                for j in range(i, end_idx):
                    nn = ckd.get_nns_by_vector(issX[j,:], neighbours, include_distances=True)
                    ckdo_ind[j,:] = nn[0]
                    ckdo_dist[j,:] = nn[1]
            
            ckdout = (ckdo_dist, ckdo_ind)
        else:
            # Original implementation
            ckdo_ind = []
            ckdo_dist = []
            for i in np.arange(issX.shape[0]):
                holder = ckd.get_nns_by_vector(issX[i,:], neighbours, include_distances=True)
                ckdo_ind.append(holder[0])
                ckdo_dist.append(holder[1])
            ckdout = (np.asarray(ckdo_dist), np.asarray(ckdo_ind))
            
    elif computation == "pynndescent":
        #build the GEX index
        ckd = pynndescent.NNDescent(gexX, metric="euclidean", n_jobs=-1, random_state=0)
        ckd.prepare()
        #query the GEX index with the ISS data
        ckdout = ckd.query(issX, k=neighbours)
        #need to reverse this to match conventions
        ckdout = (ckdout[1], ckdout[0])
    elif computation == "cKDTree":
        #build the GEX index
        ckd = scipy.spatial.cKDTree(gexX)
        #query the GEX index with the ISS data
        ckdout = ckd.query(x=issX, k=neighbours, workers=-1)
    else:
        raise ValueError("Invalid computation, must be 'annoy', 'pynndescent' or 'cKDTree'")
    return ckdout

def ckdout_to_sparse(ckdout, shape, 
                     neighbours=15
                    ):
    """
    Convert an array of KNN indices into a sparse matrix form. Return 
    the binary sparse matrix along with a copy where rows add up to 1 
    for easy mean computation.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    ckdout : tuple/list of ``np.array``
        Needs to have the KNN indices of low dimensionality cells in the 
        high dimensionality space as the second element (matching 
        ``scipy.sparse.cKDTree()`` output formatting).
    shape : list of ``int``
        The shape of the sparse matrix, low dimensionality cell count as 
        rows, high dimensionality cell count as columns.
    """
    #the indices need to be flattened, the default row-major style works
    indices = ckdout[1].flatten()
    
    #the indptr is once every neighbours, but needs an extra entry at the end
    indptr = neighbours * np.arange(shape[0]+1)
    
    #the data is ones. for now. use float32 as that's what scanpy likes as default
    data = np.ones(shape[0]*neighbours, dtype=np.float32)
    
    #construct the KNN graph!
    #need to specify the shape as there may be cells at the end that don't get picked up
    #and this will throw the dimensions off when doing matrix operations shortly
    pbs = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    
    #make a second version for means of stuff - divide data by neighbour count
    #this way each row adds up to 1 and this can be used for mean matrix operations
    pbs_means = pbs.copy()
    pbs_means.data = pbs_means.data/neighbours
    
    return pbs, pbs_means

def get_pbs_obs(iss, gex, pbs, pbs_means, 
                obs_to_take=None, 
                cont_obs_to_take=None, 
                nanmean=False, 
                obsm_fraction=False
               ):
    """
    Use the identified KNN to transfer ``.obs`` entries from the high 
    dimensionality object to the low dimensionality object. Returns a 
    majority vote/mean ``.obs`` data frame, and optionally a dictionary 
    of data frames capturing the complete fraction distribution of each 
    ``obs_to_take`` for each low dimensionality cell (for subsequent 
    ``.obsm`` insertion into the final object).
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    iss : ``AnnData``
        Low dimensionality data object with processed expression data in 
        ``.X``.
    gex : ``AnnData``
        High dimensionality data object with processed expression data in 
        ``.X``, subset to low dimensionality data object features.
    pbs : ``scipy.sparse.csr_matrix``
        Binary KNN graph, with low dimensionality cells as rows and 
        high dimensionality cells as columns.
    pbs_means : ``scipy_sparse.csr.matrix``
        KNN graph, with low dimensionality cells as rows and high 
        dimensionality cells as columns, with row values summing up to 1.
    """
    # Start the obs pool with what already resides in the ISS object
    pbs_obs = iss.obs.copy()
    
    # Possibly store all computed fractions too, will live in obsm later
    if obsm_fraction:
        pbs_obsm = {}
        
    if obs_to_take is not None:
        # Just in case a single is passed as a string
        if type(obs_to_take) is not list:
            obs_to_take = [obs_to_take]
            
        # Now we can iterate over this nicely
        # Using the logic of milopy's annotate_nhoods()
        for anno_col in obs_to_take:
            print("anno_col", anno_col)
            # Process in batches for large datasets
            if len(iss.obs_names) > 10000:
                # Create the result dataframe first
                anno_dummies = pd.get_dummies(gex.obs[anno_col])
                anno_frac = pd.DataFrame(
                    np.zeros((len(iss.obs_names), len(anno_dummies.columns))),
                    index=iss.obs_names,
                    columns=anno_dummies.columns,
                )
                
                # Process in chunks
                batch_size = 5000
                for i in range(0, pbs.shape[0], batch_size):
                    end_idx = min(i + batch_size, pbs.shape[0])
                    pbs_chunk = pbs[i:end_idx, :]
                    anno_count_chunk = pbs_chunk.dot(anno_dummies.values)
                    # Normalize counts to fractions
                    row_sums = anno_count_chunk.sum(axis=1)
                    anno_frac_chunk = anno_count_chunk / row_sums[:, np.newaxis]
                    anno_frac.iloc[i:end_idx] = anno_frac_chunk
            else:
                # Original implementation for smaller datasets
                anno_dummies = pd.get_dummies(gex.obs[anno_col])
                anno_count = pbs.dot(anno_dummies.values)
                # Convert to numpy array for division
                anno_frac = np.array(anno_count / anno_count.sum(1)[:,None])
                anno_frac = pd.DataFrame(
                    anno_frac,
                    index=iss.obs_names,
                    columns=anno_dummies.columns,
                )
                
            # Add to observations
            print("anno_frac", anno_frac)
            pbs_obs[anno_col] = anno_frac.idxmax(1)
            pbs_obs[anno_col + "_fraction"] = anno_frac.max(1)
            
            # Possibly stash full thing for obsm insertion later
            if obsm_fraction:
                pbs_obsm[anno_col] = anno_frac.copy()
                
    if cont_obs_to_take is not None:
        # Just in case a single is passed as a string
        if type(cont_obs_to_take) is not list:
            cont_obs_to_take = [cont_obs_to_take]
            
        # Process in batches for large datasets
        if len(iss.obs_names) > 10000:
            # Initialize results dataframe
            cont_obs = pd.DataFrame(
                np.zeros((len(iss.obs_names), len(cont_obs_to_take))),
                index=iss.obs_names,
                columns=cont_obs_to_take
            )
            
            # Process in chunks
            batch_size = 5000
            for i in range(0, pbs_means.shape[0], batch_size):
                end_idx = min(i + batch_size, pbs_means.shape[0])
                pbs_means_chunk = pbs_means[i:end_idx, :]
                cont_obs_chunk = pbs_means_chunk.dot(gex.obs[cont_obs_to_take].values)
                cont_obs.iloc[i:end_idx] = cont_obs_chunk
        else:
            # Original implementation for smaller datasets
            cont_obs = pbs_means.dot(gex.obs[cont_obs_to_take].values)
            cont_obs = pd.DataFrame(
                cont_obs,
                index=iss.obs_names,
                columns=cont_obs_to_take
            )
            
        # Compute a nanmean if instructed to
        if nanmean:
            # Create a helper variable with 1 if non-nan value and 0 if nan
            non_nan_mask = gex.obs[cont_obs_to_take].values.copy()
            non_nan_mask[~np.isnan(non_nan_mask)] = 1
            non_nan_mask = np.nan_to_num(non_nan_mask)
            
            # Now we can get the total non-nan counts for each cell and cont_obs
            # Process in chunks for large datasets
            if len(iss.obs_names) > 10000:
                non_nan_counts = np.zeros((len(iss.obs_names), len(cont_obs_to_take)))
                cont_obs_nanmean_values = np.zeros((len(iss.obs_names), len(cont_obs_to_take)))
                
                batch_size = 5000
                for i in range(0, pbs.shape[0], batch_size):
                    end_idx = min(i + batch_size, pbs.shape[0])
                    pbs_chunk = pbs[i:end_idx, :]
                    non_nan_counts_chunk = pbs_chunk.dot(non_nan_mask)
                    non_nan_counts[i:end_idx] = non_nan_counts_chunk
                    
                    # We can now get sums of the non-nan values for the cont_obs
                    # by filling in nans with zeroes prior to the operation
                    cont_obs_nanmean_chunk = pbs_chunk.dot(gex.obs[cont_obs_to_take].fillna(0).values)
                    cont_obs_nanmean_values[i:end_idx] = cont_obs_nanmean_chunk
            else:
                non_nan_counts = pbs.dot(non_nan_mask)
                # We can now get sums of the non-nan values for the cont_obs
                # by filling in nans with zeroes prior to the operation
                cont_obs_nanmean_values = pbs.dot(gex.obs[cont_obs_to_take].fillna(0).values)
            
            # And now we can multiply them by the weights to get the means
            # For instances where there are zero counts this is inf
            # we don't want inf, we want nan
            non_nan_weights = 1/non_nan_counts
            non_nan_weights[non_nan_weights == np.inf] = np.nan
            
            cont_obs_nanmean_values = cont_obs_nanmean_values * non_nan_weights
            
            # Store both the means and the non-nan count total
            cont_obs_nanmean = pd.DataFrame(
                np.hstack((cont_obs_nanmean_values, non_nan_counts)),
                index=iss.obs_names,
                columns=[i+"_nanmean" for i in cont_obs_to_take] + [i+"_non_nan_count" for i in cont_obs_to_take]
            )
            
        # Add continuous observations to results
        for col in cont_obs_to_take:
            pbs_obs[col] = cont_obs[col]
            
        if nanmean:
            for col in cont_obs_nanmean:
                pbs_obs[col] = cont_obs_nanmean[col]
                
    if obsm_fraction:
        return pbs_obs, pbs_obsm
    else:
        return pbs_obs

def get_pbs_X(gex_only, pbs_means, 
              round_counts=True,
              chunk_size=5000
             ):
    """
    Compute the expression of missing features in the low dimensionality 
    data as the mean of matching neighbours from the high dimensionality 
    data.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    gex_only : ``AnnData``
        High dimensionality data object with raw counts in ``.X``, subset to 
        features absent from the low dimensionality object.
    pbs_means : ``scipy_sparse.csr.matrix``
        KNN graph, with low dimensionality cells as rows and high 
        dimensionality cells as columns, with row values summing up to 1.
    """
    # Process in chunks to reduce RAM footprint
    # Initialize a CSR matrix to store results
    X = scipy.sparse.csr_matrix((0, gex_only.shape[1]))
    
    # Process chunk_size iss cells at a time
    for start_pos in range(0, pbs_means.shape[0], chunk_size):
        end_pos = min(start_pos + chunk_size, pbs_means.shape[0])
        # These are our pseudobulk definitions for this chunk
        pbs_means_sub = pbs_means[start_pos:end_pos, :]
        
        # Get the corresponding expression for the chunk
        X_sub = pbs_means_sub.dot(gex_only.X)
        
        # Round the data if requested
        if round_counts:
            X_sub.data = np.round(X_sub.data)
            X_sub.eliminate_zeros()
            
        # Store the data in the master matrix
        X = scipy.sparse.vstack([X, X_sub])
    
    return X

def knn(iss, gex, gex_only, 
        obs_to_take=None, 
        cont_obs_to_take=None, 
        nanmean=False, 
        round_counts=True, 
        chunk_size=5000,  # Reduced from 100000 for better memory management
        computation="annoy", 
        neighbours=15, 
        obsm_fraction=False, 
        obsm_pbs=False,
        memory_efficient=True
       ):
    """
    ``ip.patch()`` without the normalisation, for when custom data 
    preparation is desired.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    iss : ``AnnData``
        Low dimensionality data object with processed expression data in 
        ``.X``.
    gex : ``AnnData``
        High dimensionality data object with processed expression data in 
        ``.X``, subset to low dimensionality data object features.
    gex_only : ``AnnData``
        High dimensionality data object with raw counts in ``.X``, subset to 
        features absent from the low dimensionality object.
    memory_efficient : ``bool``
        Whether to use memory-efficient implementations
    """
    # Identify the KNN, preparing a (distances, indices) tuple
    ckdout = get_knn_indices(issX=iss.X, 
                             gexX=gex.X, 
                             computation=computation, 
                             neighbours=neighbours,
                             memory_efficient=memory_efficient
                            )
    
    # Turn KNN output into a scanpy-like graph
    # Yields a version with both ones as data and ones as row sums
    # The latter is useful for matrix operation computation of means
    pbs, pbs_means = ckdout_to_sparse(ckdout=ckdout, 
                                      shape=[iss.shape[0], gex.shape[0]], 
                                      neighbours=neighbours
                                     )
    
    # Get the annotations of the specified obs columns in the KNN
    pbs_obs = get_pbs_obs(iss=iss, 
                          gex=gex, 
                          pbs=pbs, 
                          pbs_means=pbs_means, 
                          obs_to_take=obs_to_take, 
                          cont_obs_to_take=cont_obs_to_take, 
                          nanmean=nanmean, 
                          obsm_fraction=obsm_fraction
                         )
    
    # If fractions are to be stored, this has two elements
    if obsm_fraction:
        pbs_obsm = pbs_obs[1]
        pbs_obs = pbs_obs[0]
        
    # Get the expression matrix
    X = get_pbs_X(gex_only=gex_only, 
                  pbs_means=pbs_means, 
                  round_counts=round_counts, 
                  chunk_size=chunk_size
                 )
    
    # Now we can build the object easily
    # Use a direct constructor to avoid copying X
    if obsm_pbs:
        obsm = iss.obsm.copy()
        obsm['pbs'] = pbs
        obsm['pbs_gex_obs_names'] = np.array(gex.obs_names)
    else:
        obsm = iss.obsm.copy()
    
    # Create the output object
    out = anndata.AnnData(X=X, var=gex_only.var, obs=pbs_obs, obsm=obsm)
    
    # Shove in the fractions from earlier if we need to
    if obsm_fraction:
        for anno_col in pbs_obsm:
            out.obsm[anno_col+"_fraction"] = pbs_obsm[anno_col]
            
    # Clean up large objects to help garbage collection
    del pbs, pbs_means, X
    
    return out

def patch(iss, gex, 
          min_genes=3, 
          obs_to_take=None, 
          cont_obs_to_take=None, 
          nanmean=False, 
          round_counts=True, 
          chunk_size=5000,  # Reduced from 100000 for better memory management
          computation="annoy", 
          neighbours=15, 
          obsm_fraction=False, 
          obsm_pbs=False,
          memory_efficient=True
         ):
    """
    Identify the nearest neighbours of low dimensionality observations 
    in related higher dimensionality data, approximate features absent  
    from the low dimensionality data as high dimensionality neighbour 
    means. The data is log-normalised and z-scored prior to KNN 
    inference.
    
    Input
    -----
    iss : ``AnnData``
        The low dimensionality data object, with raw counts in ``.X``.
    gex : ``AnnData``
        The high dimensionality data object, with raw counts in ``.X``.
    min_genes : ``int``, optional (default: 3)
        Passed to ``scanpy.pp.filter_cells()`` ran on the shared feature 
        space of ``iss`` and ``gex``.
    obs_to_take : ``str`` or list of ``str``, optional (default: ``None``)
        If provided, will report the most common value of the specified 
        ``gex.obs`` column(s) for the neighbours of each ``iss`` cell. 
        Discrete metadata only.
    cont_obs_to_take : ``str`` or list of ``str``, optional (default: ``None``)
        If provided, will report the average of the values of the 
        specified ``gex.obs`` column(s) for the neighbours of each 
        ``iss`` cell. Continuous metadata only.
    nanmean : ``bool``, optional (default: ``False``)
        If ``True``, will also compute an equivalent of ``np.nanmean()`` 
        for each ``cont_obs_to_take``.
    round_counts : ``bool``, optional (default: ``True``)
        If ``True``, will round the computed counts to the nearest 
        integer.
    chunk_size : ``int``, optional (default: 5000)
        Size of data chunks to process at once to reduce memory usage.
    computation : ``str``, optional (default: ``"annoy"``)
        The package supports KNN inference via annoy (specify 
        ``"annoy"``), PyNNDescent (specify ``"pynndescent"``) and scipy's 
        cKDTree (specify ``"cKDTree"``). Annoy 
        identifies approximate neighbours and runs quicker, cKDTree 
        identifies exact neighbours and is a bit slower.
    neighbours : ``int``, optional (default: 15)
        How many neighbours in ``gex`` to identify for each ``iss`` cell.
    obsm_fraction : ``bool``, optional (default: ``False``)
        If ``True``, will report the full fraction distribution of each 
        ``obs_to_take`` in ``.obsm`` of the resulting object.
    obsm_pbs : ``bool``, optional (default: ``False``)
        If ``True``, will store the identified ``gex`` neighbours for 
        each ``iss`` cell in ``.obsm['pbs']``. A corresponding vector of 
        ``gex.obs_names`` will be stored in ``.uns['pbs_gex_obs_names']``.
    memory_efficient : ``bool``, optional (default: ``True``)
        Whether to use memory-efficient implementations
    """
    # Split up the GEX into ISS features and unique features
    # Perform a quick normalisation of the ISS feature space objects
    # Keep the GEX only features as raw counts
    iss, gex, gex_only = split_and_normalise_objects(iss=iss.copy(), 
                                                    gex=gex.copy(), 
                                                    min_genes=min_genes
                                                   )
    
    # Identify the KNN and use it to approximate expression
    return knn(iss=iss, 
               gex=gex, 
               gex_only=gex_only, 
               obs_to_take=obs_to_take, 
               cont_obs_to_take=cont_obs_to_take, 
               nanmean=nanmean, 
               round_counts=round_counts, 
               chunk_size=chunk_size, 
               computation=computation, 
               neighbours=neighbours, 
               obsm_fraction=obsm_fraction, 
               obsm_pbs=obsm_pbs,
               memory_efficient=memory_efficient
              )