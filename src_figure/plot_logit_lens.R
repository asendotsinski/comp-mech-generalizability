library(ggplot2)
library(readr)
library(dplyr)
library(binom)
library(latex2exp)
library(tidyr)
library(ggpubr)
library(patchwork)

setwd("../results")

#set working directory for all the notebook knitr
print(getwd())


palette <- c("GPT2" = "#003f5c", "GPT2-medium" = "#58508d", "GPT2-large" = "#bc5090", "GPT2-xl" = "#ff6361", "Pythia-6.9b" = "#ffa600")
FACTUAL_COLOR <- "#005CAB"
COUNTERFACTUAL_COLOR <- "#E31B23"


# GPT2 Small Configurations


model <- "gpt2"
model_folder <- "gpt2_full"
n_layers <- 12
experiment <- "copyVSfact"
n_positions <- 12
positions_name <- c("-", "Subject", "2nd Subject", "3rd Subject", "Relation", "Relation Last", "Attribute*", "-", "Subject Repeat", "2nd Subject repeat", "3nd Subject repeat", "Relation repeat", "Last")
relevant_position <- c("Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last")
example_position <- c("iPhone", "was developed", "by", "Google.", "iPhone", "was developed", "by")
n_relevant_position <- 7
layer_pattern <- c(11,10,10,10,9,9)
head_pattern <- c(10,0,7,10,6,9)
layer_pattern <- c(11,10,10,10,9,9)
head_pattern <- c(10,0,7,10,6,9)
factual_heads <- c(c(11,10),c(10,7))
factual_heads_layer <- c(11,10)
factual_heads_head <- c(10,7)
AXIS_TITLE_SIZE <- 60
AXIS_TEXT_SIZE <- 50
HEATMAP_SIZE <- 10


# Pythia-6.9b Small Configurations

# model <- "pythia-6.9b"
# model_folder <- "pythia-6.9b_full"
# n_layers <- 32
# experiment <- "copyVSfact"
# n_positions <- 12
# positions_name <- c("-", "Subject", "2nd Subject", "3rd Subject", "Relation", "Relation Last", "Attribute*", "-", "Subject Repeat", "2nd Subject repeat", "3nd Subject repeat", "Relation repeat", "Last")
# relevant_position <- c("Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last")
# n_relevant_position <- 7
# layer_pattern <- c(10,10,15 ,17,17,19,19,20,20,21,23)
# head_pattern <-  c(1,27, 17, 14,28,20,31,2, 18,8,25)
# factual_heads <- c(c(11,10),c(10,7))
# AXIS_TITLE_SIZE <- 60
# AXIS_TEXT_SIZE <- 50
# HEATMAP_SIZE <- 10

# Load functions:



LOW_COLOR <- FACTUAL_COLOR
HIGH_COLOR <- COUNTERFACTUAL_COLOR

#LOW_COLOR <- "#1a80bb" # factual
#HIGH_COLOR <- "#a00000" # counterfactual
create_heatmap_base <- function(data, x, y, fill, midpoint = 0, text=FALSE) {
  # Convert strings to symbols for tidy evaluation
  x_sym <- rlang::sym(x)
  y_sym <- rlang::sym(y)
  fill_sym <- rlang::sym(fill)
  if (text==TRUE){
    p<- ggplot(data, aes(!!x_sym, !!y_sym, fill = !!fill_sym)) +
      geom_tile(colour = "grey") +
      scale_fill_gradient2(low = LOW_COLOR, mid = "white", high = HIGH_COLOR, midpoint = midpoint) +
      theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1)) +
      geom_text(aes(label = sprintf("%.2f", !!fill_sym)), color = "black", size = HEATMAP_SIZE)+
      labs(x = x, y = y)
  }else{
  p<- ggplot(data, aes(!!x_sym, !!y_sym, fill = !!fill_sym)) +
    geom_tile(colour = "grey") +
   scale_fill_gradient2(low = LOW_COLOR, mid = "white", high = HIGH_COLOR) +
    theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1)) +
   # geom_text(aes(label = sprintf("%.2f", !!fill_sym)), color = "black", size = HEATMAP_SIZE)+
    labs(x = x, y = y) 
  }
  return(p)
}


## Logit Lens - Residual Stream

### Functions


create_heatmap <- function(data, x, y, legend, fill, high_color) {
  p <- create_heatmap_base(data, x, y, fill) +
    scale_fill_gradient2(low = "white", mid = "white", high = high_color, limits=c(-1,17), name = legend) +
    #scale_fill_gradient2(low = "black", mid = "white", high = high_color, name = "Logit") +
    #scale_fill_gradient2(low = "black", mid= "white", high = high_color, name = "Logit") +
    
    theme_minimal() +
    #addforce to have all the labels
    #scale_x_continuous(breaks = seq(0,n_layers-1 ,1)) +
    scale_y_continuous(breaks = seq(0,n_relevant_position -1,1), labels = relevant_position,
                      sec.axis = dup_axis(name = "", labels =  example_position)) +
    scale_y_reverse(breaks = seq(0,n_relevant_position -1,1), labels = relevant_position,
                    sec.axis = dup_axis(name = "", labels = example_position))+
    labs(x = "Layer", y = "")+
    #fix intenxity of fill
    theme(
      axis.text.x = element_text(size=AXIS_TEXT_SIZE),
      axis.text.y = element_text(size=AXIS_TEXT_SIZE,),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = AXIS_TITLE_SIZE),
      axis.title.y = element_text(size = AXIS_TITLE_SIZE),
      legend.text = element_text(size = 40),
      legend.title = element_text(size = AXIS_TEXT_SIZE),
      #remove the legend\
      legend.position = "bottom",
      #increase the legend size
      legend.key.size = unit(2.5, "cm"),
      # move the y ticks to the right
    ) 
  return(p)
}



print(getwd())


### Load and process data


#print current working directory
data <- read.csv(sprintf("%s/logit_lens/%s/logit_lens_data.csv", experiment, model_folder))
number_of_position <- max(as.numeric(data$position))
data_resid_post <- data %>% filter(grepl("resid_post", component))
data_resid_post$position_name <- positions_name[data_resid_post$position + 1]
#filter just the relevant positions
# data_resid_post <- data_resid_post %>% filter(position == 1 | position==4 | position == 5 | position== 6 | position==8 |  position == 11 | position== 12)
data_resid_post <- data_resid_post %>% filter(position == 1 | position==4 | position == 5 | position== 6 | position==8 |  position == 12 | position== 13)
unique_positions <- unique(data_resid_post$position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_resid_post$mapped_position <- unname(position_mapping[as.character(data_resid_post$position)])


### Residual stram heatmaps


p_fact <- create_heatmap(data_resid_post, "layer", "mapped_position", bquote("Logit of " * italic(t[fact])), "mem",  FACTUAL_COLOR)
p_copy <- create_heatmap(data_resid_post, "layer", "mapped_position", bquote("Logit of " * italic(t[cofa])), "cp",  COUNTERFACTUAL_COLOR)
p_fact
p_copy


# Save it:

ggsave(sprintf("paper_plots/%s_%s_residual_stream/resid_post_mem.pdf", model, experiment), p_fact, width = 50, height = 32, units = "cm", create.dir = TRUE)
ggsave(sprintf("paper_plots/%s_%s_residual_stream/resid_post_cp.pdf", model, experiment), p_copy, width = 50, height = 32, units = "cm")


### Residual streams - lineplot index


data_resid_post_altered <- data_resid_post %>% filter(position == 6)
data_resid_post_2_subject <- data_resid_post %>% filter(position == 8)
data_resid_post_last <- data_resid_post %>% filter(position ==13)
p_logit <-ggplot(data_resid_post_last, aes(x=layer))+
  #last
  geom_line(aes(y=mem, color="mem"),size=4,  alpha=0.8 )+
  geom_point(aes(y=mem, color="mem"),size=6, alpha=0.8)+
  geom_line(aes(y=cp, color="cp"),size=4,  alpha=0.8)+
  geom_point(aes(y=cp, color="cp"),size=6,  alpha=0.8)+
  scale_color_manual(values = c("mem" = FACTUAL_COLOR, "cp" = COUNTERFACTUAL_COLOR, "cp_alt"="darkblue", "mem_subj"="darkred"), labels=c("cp"= "Counterfactual Token","mem"="Factual Token", "cp_alt"= "Counterfactual Attribute","mem_subj"="Factual 2nd Subject")) +
  labs(y= "Logit in the Last Position", x="Layer", color="")+
  theme_minimal()+
 scale_x_continuous(breaks = seq(0,n_layers-1,1)) +
  scale_y_continuous(limits = c(0,17))+
 # scale_y_log10()+
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE,),
    #remove background grid
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = AXIS_TEXT_SIZE),
    legend.title = element_text(size = AXIS_TEXT_SIZE),
    #remove the legend\
    legend.position = "bottom",
    #increase the legend size
    legend.key.size = unit(2, "cm"),
    aspect.ratio = 7/10,
    panel.border = element_rect(colour = "grey", fill=NA, size=1),
    # move the y ticks to the right
  ) + guides(color = guide_legend(ncol = 2, nrow=1))
p_logit
ggsave(sprintf("paper_plots/%s_%s_residual_stream/resid_post_all_linelogit_line.pdf", model, experiment), p_logit, width = 50, height = 30, units = "cm")


# save
# p <- create_heatmap(data_resid_post, "layer", "mapped_position", "ratio",  "darkgreen")
# p <- create_heatmap(data_resid_post, "layer", "mapped_position", bquote("Logit of " * italic(t[cofa])), "cp",  COUNTERFACTUAL_COLOR)
# ggsave(sprintf("paper_plots/%s_%s_residual_stream/resid_post_cp_index.pdf", model, experiment), p, width = 50, height = 30, units = "cm")


### Residual streams - lineplot logit


data_resid_post_altered <- data_resid_post %>% filter(position == 6)
data_resid_post_2_subject <- data_resid_post %>% filter(position == 8)
data_resid_post_last <- data_resid_post %>% filter(position ==12)

p_idx<-ggplot(data_resid_post_altered, aes(x=layer))+
  geom_line(aes(y=mem_idx, color="mem"),size=4,  alpha=0.8 )+
  geom_point(aes(y=mem_idx, color="mem"),size=6, alpha=0.8)+
  geom_line(aes(y=cp_idx, color="cp"),size=4,  alpha=0.8)+
  geom_point(aes(y=cp_idx, color="cp"),size=6,  alpha=0.8)+
  scale_color_manual(values = c("mem" = FACTUAL_COLOR, "cp" = COUNTERFACTUAL_COLOR), labels=c( "Altered Token", "Factual Token")) +
  labs(y= bquote("Rank"), x="Layer", color="")+
  theme_minimal()+
  #scale_x_continuous(breaks = seq(0,n_layers,1)) +
  scale_y_log10()+
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE,),
    #remove background grid
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = AXIS_TEXT_SIZE),
    legend.title = element_text(size = AXIS_TEXT_SIZE),
    #remove the legend\
    legend.position = "bottom",
    #increase the legend size
    legend.key.size = unit(2, "cm"),
    # move the y ticks to the right
  )
p_idx


model
# Save
ggsave(sprintf("paper_plots/%s_%s_residual_stream/resid_post_index.pdf", model, experiment), p_idx, width = 50, height = 30, units = "cm")


### Multiple plots
#### With logit

spacer <- plot_spacer()
p_logit <- p_logit + theme(aspect.ratio = 7/10, legend.position = c(0.5,-0.2))
p <- (p_fact / p_copy) | spacer | p_logit
p <- p + plot_layout(widths = c(0.8, 0.4, 1.3))
ggsave(sprintf("paper_plots/%s_%s_residual_stream/resid_post_all_linelogit.pdf", model, experiment), p, width = 100, height = 50, units = "cm")
#ggsave("resid_post_all_linelogit_new.pdf", p, width = 100, height = 50, units = "cm")

#### With index

spacer <- plot_spacer()
p_idx <- p_idx + theme(aspect.ratio = 7/10, legend.position = c(0.5,-0.2))
p <- (p_fact / p_copy) | spacer | p_idx
p <- p + plot_layout(widths = c(0.8, 0.3, 1.3))
ggsave(sprintf("paper_plots/%s_%s_residual_stream/resid_post_all_lineindex.pdf", model, experiment), p, width = 100, height = 50, units = "cm")