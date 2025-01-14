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


## Logit Attribution
### Functions

create_heatmap <- function(data, x, y, fill, head=FALSE) {
  if(head){
    scale_x <- scale_x_discrete(breaks = seq(0,n_layers,1)) 
    angle = 0
  } else {
    scale_x <- scale_x_discrete(breaks = seq(0, n_positions,1), labels = positions_name)
    angle = 90
  }
  print(n_positions)
  p <- create_heatmap_base(data, x, y, fill) +
    theme_minimal() +
    #addforce to have all the labels
    scale_y_discrete(breaks = seq(0,n_layers,1)) +
    scale_x +
    labs(fill = "Logit Diff") +
    theme(
      axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = angle),
      axis.text.y = element_text(size=AXIS_TEXT_SIZE),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = AXIS_TITLE_SIZE),
      axis.title.y = element_text(size = AXIS_TITLE_SIZE),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = 30),
      #remove the legend\
      legend.position = "bottom",
      # increase the size of the legend
      legend.key.size = unit(2, "cm"),
      # move the y ticks to the right
    )
  return(p)
}

### Load and process data

data <- read.csv(sprintf("%s/logit_attribution/%s/logit_attribution_data.csv", experiment, model_folder))

### Head

data_head <- data %>% filter(grepl("^L[0-9]+H[0-9]+$", label))
number_of_position <- max(as.numeric(data_head$position))
print(number_of_position)
## filter to have just position 12
data_head_ <- data_head %>% filter(position == number_of_position)
# for each row split L and H and create a new column for each
data_head_ <- data_head_ %>% separate(label, c("layer", "head"), sep = "H")
#renominating the columns layer and head to Layer and Head
#remove L from layer
data_head_$layer <- gsub("L", "", data_head_$layer)

max_layer <- max(as.numeric(data_head_$layer))
max_head <- max(as.numeric(data_head_$head))
data_head_$layer <- factor(data_head_$layer, levels = c(0:max_layer))
data_head_$head <- factor(data_head_$head, levels = c(0:max_head))
colnames(data_head_)[1] <- "Layer"
colnames(data_head_)[2] <- "Head"
data_head_$diff_mean <- -data_head_$diff_mean



p <- create_heatmap_base(data_head_, "Layer", "Head", "diff_mean") +
  theme_minimal() +
  #addforce to have all the labels
  scale_y_discrete(breaks = seq(0,n_layers,1)) +
  scale_x_discrete(breaks = seq(0,n_layers,1))  +
  labs(fill = expression(Delta[cofa])) +
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE -8, angle = 0),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE-8),
    #remove background grid
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = 40),
    legend.title = element_text(size = 70),
    #remove the legend\
    legend.position = "bottom",
    # increase the size of the legend
    legend.key.size = unit(2.5, "cm"),
    # move the y ticks to the right
  )
ggsave(sprintf("paper_plots/%s_%s_logit_attribution/logit_attribution_head_position%s.pdf", model, experiment, number_of_position), p, width = 60, height = 60, units = "cm", create.dir=TRUE)



# Then perform some computation on the heads:

### count the impact of the positive head ###
#sum all the negative values
factual_impact <- data_head_ %>% group_by(Layer) %>% summarise(positive_impact = sum(diff_mean[diff_mean < 0]))
# sum also across layers
factual_impact <- factual_impact %>% summarise(positive_impact = sum(positive_impact))
l10h7 <- data_head_ %>% filter(Layer == 10, Head == 7)
l10h7 <- l10h7$diff_mean
l11h10 <- data_head_ %>% filter(Layer == 11, Head == 10)
l11h10 <- l11h10$diff_mean

l10h7 <- 100 * l10h7 / sum(factual_impact)
print(l10h7)
l11h10 <- 100 * l11h10 / sum(factual_impact)
print(l11h10)



### MLP and Attention Barplot

data_mlp <- data %>% filter(grepl("^[0-9]+_mlp_out$", label))
data_mlp <- data_mlp %>% separate(label, c("layer"), sep = "_mlp_out")
max_position <- max(as.numeric(data_mlp$position))
data_mlp <- data_mlp %>% filter(position == max_position)
data_attn <- data %>% filter(grepl("^[0-9]+_attn_out$", label))
data_attn <- data_attn %>% separate(label, c("layer"), sep = "_attn_out")
max_position <- max(as.numeric(data_mlp$position))
data_attn <- data_attn %>% filter(position == max_position)
#merge the two dataframe
data_barplot <- data_mlp
data_barplot$attn_dif <- data_attn$diff_mean
data_barplot$attc_cp <- data_attn$cp_mean
data_barplot$attc_mem <- data_attn$mem_mean
#rename columns diff_mean to mlp_dif
data_barplot <- data_barplot %>% rename("mlp_dif" = diff_mean)
data_barplot <- data_barplot %>% rename("mlp_cp" = cp_mean)
data_barplot <- data_barplot %>% rename("mlp_mem" = mem_mean)

#pivoting attn in order to plot mem and cp in the same barplot
data_attn <- data_barplot %>% pivot_longer(cols = c("attc_cp", "attc_mem"), names_to = "Block", values_to = "value")
data_mlp <- data_barplot %>% pivot_longer(cols = c("mlp_cp", "mlp_mem"), names_to = "Block", values_to = "value")
#modify mlp_cp and mlp_mem to Altered and Factual
data_mlp$Block <- gsub("mlp_", "", data_mlp$Block)
data_mlp$Block <- gsub("cp", "Altered", data_mlp$Block)
data_mlp$Block <- gsub("mem", "Factual", data_mlp$Block)
data_attn$Block <- gsub("attc_", "", data_attn$Block)
data_attn$Block <- gsub("cp", "Altered", data_attn$Block)
data_attn$Block <- gsub("mem", "Factual", data_attn$Block)
data_barplot <- data_barplot %>% rename("MLP Block" = mlp_dif )
data_barplot <- data_barplot %>% rename("Attention Block" = attn_dif)


data_barplot$`MLP Block` <- -data_barplot$`MLP Block`
data_barplot$`Attention Block` <- -data_barplot$`Attention Block`

data_barplot$layer <- as.numeric(data_barplot$layer) 

#### MLP

ggplot(data_barplot, aes(x = as.numeric(layer), y = `MLP Block`, fill = "MLP Block")) +
  geom_col(position = position_dodge(), color="black", size=1) +
  labs(x = "Layer", y = expression(Delta[cofa]), fill = "") + # Naming the legend
  theme_minimal() +
  scale_fill_manual(values = c("MLP Block" = "#bc5090")) + # Assigning color to the "MLP Block"
  #scale_y_continuous(limits = c(-1, 1.5)) +
  scale_x_continuous(breaks= seq(0, n_layers-1, 1), labels = as.character(seq(0,n_layers-1,1))) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 60),
    axis.text.y = element_text(size = 60),
    axis.title.x = element_text(size = 60),
    axis.title.y = element_text(size = 70),
    legend.text = element_text(size = 50),
    legend.title = element_text(size = 55),
    legend.position = "top"
  ) +
  guides(fill = guide_legend(ncol = 2.5)) # Adjusting the legend

# ggsave("mlp_block_norm.pdf", width = 50, height = 30, units = "cm")
ggsave(sprintf("paper_plots/%s_%s_logit_attribution/mlp_block_norm.pdf", model, experiment, max_position), width = 50, height = 30, units = "cm")


#### Attention Out

ggplot(data_barplot, aes(x = as.numeric(layer), y = `Attention Block`, fill = "Attention Block")) +
  geom_col(position = position_dodge(), color="black",size=1) +
  labs(x = "Layer", y = expression(Delta[cofa]), fill = "") + # Naming the legend
  theme_minimal() +
  scale_fill_manual(values = c("Attention Block" = "#ffa600")) + # Assigning color to the "MLP Block"
  #scale_y_continuous(limits = c(-1, 1.5)) +
  scale_x_continuous(breaks= seq(0, n_layers-1, 1), labels = as.character(seq(0,n_layers-1,1))) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 60),
    axis.text.y = element_text(size = 60),
    axis.title.x = element_text(size = 60),
    axis.title.y = element_text(size = 70),
    legend.text = element_text(size = 50),
    legend.title = element_text(size = 55),
    legend.position = "top"
  ) +
  guides(fill = guide_legend(ncol = 2.5)) # Adjusting the legend

# ggsave("attn_block_norm.pdf", width = 50, height = 30, units = "cm")
ggsave(sprintf("paper_plots/%s_%s_logit_attribution/attn_block_norm.pdf", model, experiment, max_position), width = 50, height = 30, units = "cm")



#### HeatMaps
##### MLP

data_mlp <- data %>% filter(grepl("^[0-9]+_mlp_out$", label))
data_mlp <- data_mlp %>% separate(label, c("layer"), sep = "_mlp_out")
max_layer <- max(as.numeric(data_mlp$layer))
max_position <- max(as.numeric(data_mlp$position))
#create layer column

data_mlp$layer <- factor(data_mlp$layer, levels = c(0:max_layer))
data_mlp$position <- factor(data_mlp$position, levels = c(0:max_position))

colnames(data_mlp)[1] <- "Layer"
colnames(data_mlp)[2] <- "Position"

data_mlp <- data_mlp %>% filter(Position == 1 | Position==4 | Position == 5 | Position== 6 | Position==8 |  Position == 11 | Position== 12)
unique_positions <- unique(data_mlp$Position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_mlp$mapped_position <- unname(position_mapping[as.character(data_mlp$Position)])
data_mlp$Layer <- as.numeric(data_mlp$Layer) +1

relevant_position <- c("Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last")
n_relevant_position <- 7
data_mlp$diff_mean <- -data_mlp$diff_mean
p <- create_heatmap_base(data_mlp, "Layer", "mapped_position", "diff_mean") +
  theme_minimal() +
  #addforce to have all the labels
  scale_x_continuous(breaks = seq(1,max_layer+1,1), labels=as.character(seq(1,max_layer+1,1))) +
  scale_y_reverse(breaks = seq(0,n_relevant_position-1, 1), labels=relevant_position) +
  labs(fill = "Logit Diff", y="") +
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE),
    #remove background grid
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = 35),
    legend.title = element_text(size = AXIS_TITLE_SIZE),
    #remove the legend\
    legend.position = "bottom",
    # increase the size of the legend
    legend.key.size = unit(3, "cm"),
    # move the y ticks to the right
  )

p
ggsave(sprintf("paper_plots/%s_%s_logit_attribution/logit_attribution_mlp_out.pdf", model, experiment), p, width = 50, height = 50, units = "cm")




##### Attn out

#filter position f"{i}_mlp_out"
data_attn <- data %>% filter(grepl("^[0-9]+_attn_out$", label))
data_attn <- data_attn %>% separate(label, c("layer"), sep = "_attn_out")
max_layer <- max(as.numeric(data_attn$layer))
max_position <- max(as.numeric(data_attn$position))
#create layer column

data_attn$layer <- factor(data_attn$layer, levels = c(0:max_layer))
data_attn$position <- factor(data_attn$position, levels = c(0:max_position))

colnames(data_attn)[1] <- "Layer"
colnames(data_attn)[2] <- "Position"
data_attn <- data_attn %>% filter(Position == 1 | Position==4 | Position == 5 | Position== 6 | Position==8 |  Position == 11 | Position== 12)
unique_positions <- unique(data_attn$Position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_attn$mapped_position <- unname(position_mapping[as.character(data_attn$Position)])
data_attn$Layer <- as.numeric(data_attn$Layer) -1

relevant_position <- c("Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last")
n_relevant_position <- 7
data_attn$diff_mean <- -data_attn$diff_mean

p <- create_heatmap_base(data_attn, "Layer", "mapped_position", "diff_mean") +
  theme_minimal() +
  #addforce to have all the labels
  scale_x_continuous(breaks = seq(1,max_layer+1,1), labels=as.character(seq(1,max_layer+1,1))) +
  scale_y_reverse(breaks = seq(0,n_relevant_position-1, 1), labels=relevant_position) +
  labs(fill = "Logit Diff", y="") +
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE),
    #remove background grid
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = 35),
    legend.title = element_text(size = AXIS_TITLE_SIZE),
    #remove the legend\
    legend.position = "bottom",
    # increase the size of the legend
    legend.key.size = unit(3, "cm"),
    # move the y ticks to the right
  )

p
ggsave(sprintf("paper_plots/%s_%s_logit_attribution/logit_attribution_attn_out.pdf", model, experiment), p, width = 50, height = 50, units = "cm")