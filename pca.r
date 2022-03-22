library(ade4)
library("factoextra")
library(ggpubr)
library(gridExtra)
library(hexbin)
library(matrixStats)
library(flexclust)
library("viridis")


comp_freqs <- function(em_runs, n_cl, in_freq_path, in_weights_path, profiles) {
  # compare avg. cluster freqs and em cluster freqs (simulations)
  print(c(length(em_runs), strtoi(n_cl)))
  par(mfrow=c(length(em_runs), strtoi(n_cl)))
  for (em in em_runs) { # for each plot 
    filename <- paste(n_cl, "cl_em_", em, "_", level, ".csv", sep = "")
    sim_path <- paste(in_freq_path, "/", filename, sep = "")
    print(filename)
    
    # load simulations and profile weights 
    aafreqs_sim_genes <- read.csv(sim_path)
    for (cl in seq(strtoi(n_cl))) {
      if (strtoi(n_cl) > 1) { 
        pro_w <- read.csv(paste(in_weights_path, "/cl", cl, "_pro_weights_", em, 
                                ".csv", sep = ""), header = FALSE)
        
        cl_freqs <- t(as.matrix(profiles) %*% as.vector(t(pro_w)))
        mean_freq_cl <- colMeans(as.matrix(aafreqs_sim_genes[aafreqs_sim_genes$cl == 1, 1:20]))
        mae <- mean(abs(cl_freqs - mean_freq_cl))
        sort_order <- order(mean_freq_cl)
        plot(cl_freqs[sort_order], mean_freq_cl[sort_order], 
             xlab = paste("EM", em, "cl.", cl, "- mae :", round(mae, 4)), 
             ylab = paste("Cl.", cl, "(avg. sim.)"))
        lines(mean_freq_cl[sort_order], mean_freq_cl[sort_order], type = 'l')
      }
    }
  }
}

z_score <- function(x) {
  return ((x - mean(x)) / sd(x))} 

n_cols <- 3
all_cols <- viridis(n_cols)

n_div <- 4
n_runs <- 5
n_cls <- c('3', '1')
em_runs <- c(sprintf("%s",seq(5)), 'avg')
level = 'alns'

path2script <- dirname(rstudioapi::getSourceEditorContext()$path)

out_path <- paste(path2script, "/results/pca", sep = "")
in_freq_path <- paste(path2script, "/data/freq_samples", sep = "")
in_weights_path <- paste(path2script, "/results/profiles_weights/3cl", sep = "")
in_profiles_path <- paste(path2script, "/results/profiles_weights/263-hogenom-profiles.tsv", sep = "")

profiles <- read.csv(in_profiles_path, sep = '\t', header = FALSE)
# aa_classes <- read.csv("~/data/aa_classes.csv")

# load empirical frequencies
aafreqs_real_genes <- read.csv(paste(in_freq_path, "/fasta_no_gaps_alns.csv", sep = ""))

# PCA
pca.real.genes <- dudi.pca(aafreqs_real_genes[,1:20], scannf = FALSE, nf = 4)

# exclude 0.2% at each side (in axes 3 and 4)
mean_pcs34 <- apply(pca.real.genes$li[,3:4], c(2), mean)
dist_center_pcs34 <- dist2(pca.real.genes$li[,3:4], mean_pcs34, method = "euclidean", p=2)
inds <- dist_center_pcs34 < quantile(dist_center_pcs34, 0.995) & dist_center_pcs34 > quantile(dist_center_pcs34, 0.005)
pca.real.genes.filt <- pca.real.genes[inds,]

# keep every other seq (n_div times)
if (n_div > 0) {
  for (x in seq(n_div)) {
    pca.real.genes.filt <- pca.real.genes.filt[seq(1, 
                                                   nrow(pca.real.genes.filt$li), 2),]
  }  
}
n_genes <- nrow(pca.real.genes.filt$li)


# simulations on real data 
for (n_cl in n_cls) {
  for (em in em_runs) { # for each plot 
    pcslim = cbind(colMins(as.matrix(pca.real.genes$li)), 
                   colMaxs(as.matrix(pca.real.genes$li)))
    
    filename <- paste(n_cl, "cl_em_", em, "_", level, ".csv", sep = "")
    sim_path <- paste(in_freq_path, "/", filename, sep = "")
    print(filename)
    
    # load simulations and profile weights 
    aafreqs_sim_genes <- read.csv(sim_path)
    
    grps_cl <- c()
    sim.coord.cls <- list()
    cls_freqs.coord <- data.frame(matrix(0, strtoi(n_cl), 4))
    for (cl in seq(strtoi(n_cl))) {
      if (strtoi(n_cl) > 1) { 
        pro_w <- read.csv(paste(in_weights_path, "/cl", cl, "_pro_weights_", em, 
                                ".csv", sep = ""), header = FALSE)
        
        cl_freqs <- t(as.matrix(profiles) %*% as.vector(t(pro_w))) 
        cls_freqs.coord[cl,] <- suprow(pca.real.genes, cl_freqs)$lisup
        grps_cl <- c(grps_cl, paste("cluster", cl))
      }
      # predict coordinates for simulations
      sim.coord.cl <- suprow(pca.real.genes,
                             aafreqs_sim_genes[aafreqs_sim_genes$cl == cl, 
                                               1:20])$lisup
      # keep every other seq
      if (n_div > 0) {
        for (x in seq(n_div)) {
          sim.coord.cl <- sim.coord.cl[seq(1, nrow(sim.coord.cl), 2),]
        }  
      }
      
      # get/update limits 
      pcslim.sim = cbind(colMins(as.matrix(sim.coord.cl)), 
                         colMaxs(as.matrix(sim.coord.cl)))
      pcslim <- cbind(rowMins(as.matrix(cbind(pcslim.sim[,1], 
                                              pcslim[,1]))), 
                      rowMaxs(as.matrix(cbind(pcslim.sim[,2], 
                                              pcslim[,2]))))
      sim.coord.cls <- append(sim.coord.cls, list(sim.coord.cl))
    
    }
    
    # generate pca plot 
    cl_cols <- all_cols[1:(strtoi(n_cl))]
    cl_shape <- c(9, 10, 12)
    
    pca.real.genes.plot.ax12 <- fviz_pca_ind(pca.real.genes.filt, axes = c(1, 2),
                                             label = "empirical",
                                             alpha.ind = 1,                            
                                             title = filename) +          
      geom_point() + scale_x_continuous(limits=pcslim[1,]) +
      scale_y_continuous(limits=pcslim[2,]) + 
      theme(plot.title = element_text(size=6))
    
    pca.real.genes.plot.ax34 <- fviz_pca_ind(pca.real.genes.filt, axes = c(3, 4),
                                             label = "empirical",
                                             alpha.ind = 1,                            
                                             title = filename,
                                             #col.ind = rep("Empirical", n_genes)
                                             ) +          
      geom_point() + scale_x_continuous(limits=pcslim[3,]) +
      scale_y_continuous(limits=pcslim[4,]) + 
      theme(plot.title = element_text(size=6))
    
    for (cl in seq(strtoi(n_cl))) {
      # add simulations from cluster x to plot
      pca.real.genes.plot.ax12 <- fviz_add(pca.real.genes.plot.ax12, 
                                           sim.coord.cls[[cl]],
                                           axes = c(1, 2),
                                           addlabel = FALSE, 
                                           color = cl_cols[cl],
                                           shape = cl_shape[cl],)
      pca.real.genes.plot.ax34 <- fviz_add(pca.real.genes.plot.ax34, 
                                           sim.coord.cls[[cl]],
                                           axes = c(3, 4),
                                           color = cl_cols[cl],
                                           shape = cl_shape[cl],
                                           addlabel = FALSE)
    }
    if (strtoi(n_cl) > 1) {
      for (cl in seq(strtoi(n_cl))) {
        # add theoretical cluster frequencies 
        pca.real.genes.plot.ax12 <- fviz_add(pca.real.genes.plot.ax12, 
                                             cls_freqs.coord[cl,],
                                             axes = c(1, 2),
                                             addlabel = FALSE, 
                                             color = "red", 
                                             shape=cl_shape[cl], pointsize=4)
        pca.real.genes.plot.ax34 <- fviz_add(pca.real.genes.plot.ax34, 
                                             cls_freqs.coord[cl,],
                                             axes = c(3, 4),
                                             color = "red", 
                                             addlabel = FALSE,
                                             shape=cl_shape[cl], pointsize=4)
      }  
    }
    
    # save plots 
    basename <- strsplit(filename, '.csv')
    # real versus sim - pca
    out_path_pca = paste(out_path, "/", paste(basename, "_pca12", ".png", 
                                              sep = ""),
                         sep = "")
    png(out_path_pca, width = 846, height = 438, res = 110)
    print(pca.real.genes.plot.ax12)
    dev.off()
    out_path_pca = paste(out_path, "/", paste(basename, "_pca34", ".png", 
                                              sep = ""),
                         sep = "")
    png(out_path_pca, width = 846, height = 438, res = 110)
    print(pca.real.genes.plot.ax34)
    dev.off()
  }
}

'''
# density
out_path_hm_sim = paste(out_path, paste(basename, "_hmsim", ".png", 
                                         sep = ""), sep = "")
out_path_hm_real = paste(out_path, paste(basename, "_hmreal", ".png", 
                                          sep = ""), sep = "")

xbnds <- range(pcs.sim.genes[,1], pcs.real.genes[,1])
ybnds <- range(pcs.sim.genes[,2], pcs.real.genes[,2])
sim_hm <- hexbin(pcs.sim.genes, xbins = 10, xbnds=xbnds,ybnds=ybnds)
real_hm <- hexbin(pcs.real.genes, xbins = 10, xbnds=xbnds,ybnds=ybnds)

png(out_path_hm_sim, width = 946, height = 639)
print(plot(sim_hm))
dev.off()
png(out_path_hm_real, width = 946, height = 639)
plot(real_hm)
dev.off()
'''

'''
# profiles on real data
profiles.coord <- suprow(pca.real.genes, t(profiles))$lisup
# real pca plots
pca.real.genes.plot.ax12 <- fviz_pca_ind(pca.real.genes.filt, axes = c(1, 2),
                                         label = "empirical",
                                         alpha.ind = 1,                            
                                         title = strsplit(basename(in_profiles_path), 
                                                          ".tsv")) + 
  theme(plot.title = element_text(size=8))
pca.real.genes.plot.ax12 <- fviz_add(pca.real.genes.plot.ax34, 
                                     profiles.coord,
                                     axes = c(1, 2), addlabel = FALSE)

pca.real.genes.plot.ax34 <- fviz_pca_ind(pca.real.genes.filt, axes = c(3, 4),
                                         label = "empirical",
                                         alpha.ind = 1,                            
                                         title = strsplit(basename(in_profiles_path), 
                                                          ".tsv")
                                         #col.ind = rep("Empirical", n_genes)
) + 
  theme(plot.title = element_text(size=8))
pca.real.genes.plot.ax34 <- fviz_add(pca.real.genes.plot.ax34,  
                                     profiles.coord,
                                     axes = c(1, 2), addlabel = FALSE)
# save plots 
basename <- strsplit(basename(in_profiles_path), ".tsv")
# real versus sim - pca
out_path_pca = paste(out_path, "/", paste(basename, "_pca12", ".png", 
                                          sep = ""),
                     sep = "")
png(out_path_pca, width = 846, height = 438, res = 110)
print(pca.real.genes.plot.ax12)
dev.off()
out_path_pca = paste(out_path, "/", paste(basename, "_pca34", ".png", 
                                          sep = ""),
                     sep = "")
png(out_path_pca, width = 846, height = 438, res = 110)
print(pca.real.genes.plot.ax34)
dev.off()

'''