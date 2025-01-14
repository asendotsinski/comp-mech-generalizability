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

## Head Pattern
data <- read.csv(sprintf("%s/head_pattern/%s/head_pattern_data.csv", experiment, model_folder))

### Last Position
data_filtered <- data %>% filter(source_position == 12)
pattern_df <- data.frame(layer = layer_pattern, head = head_pattern)

data_final <- data_filtered %>% 
  inner_join(pattern_df, by = c("layer", "head"))
# Step 3: Prepare the data for plotting
data_final$y_label <- paste("Layer ", data_final$layer, " | Head ", data_final$head, sep="")
#filter just the relevant positions
data_final <- data_final %>% filter(dest_position == 1 | dest_position==4 | dest_position == 5 | dest_position== 6 | dest_position==8 |  dest_position == 11 | dest_position== 12)
unique_positions <- unique(data_final$dest_position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_final$mapped_position <- unname(position_mapping[as.character(data_final$dest_position)])
# Create and plot the heatmap
data_final <- data_final %>%
  mutate(color = ifelse((
    y_label =="Layer 10 | Head 27" | 
    y_label=="Layer 17 | Head 28" |
    y_label=="Layer 20 | Head 18" |
    y_label=="Layer 21 | Head 8"
    ), "Target", "Other")) # Add color column
#gpt2 
# data_final <- data_final %>%
#   mutate(color = ifelse((y_label =="Layer 10 | Head 7" | y_label=="Layer 11 | Head 10"), "Target", "Other")) # Add color column
#pythia
data_final <- data_final %>%
 mutate(color = ifelse((y_label =="Layer 21 | Head 8" | y_label=="Layer 20 | Head 18" | y_label=="Layer 17 | Head 28" |  y_label=="Layer 10 | Head 27"), "Target", "Other")) # Add color column

library(ggnewscale) # for using new color scales within the same plot
# Your original plot for 'Other'




max_positions <- max(data_final$mapped_position)
tile_width <- 1 
tile_height <- 1 


heatmap_plot <- ggplot(data_final %>% filter(color == "Target"), aes(x = mapped_position, y = y_label, fill = value)) +
  geom_tile(colour = "grey", width = tile_width, height = tile_height) +
  scale_x_continuous(breaks = seq(0, length(relevant_position) - 1,1), labels = relevant_position) +
  scale_y_discrete(limits = unique(data_final$y_label)) +
  scale_fill_gradient(low = "white", high = FACTUAL_COLOR, limits=range(data_final$value)) +
  labs(fill = "Attention\nScore:") +
  theme_minimal() +
  new_scale_fill() + # This tells ggplot to start a new fill scale
  geom_tile(data = data_final %>% filter(color == "Other"), aes(x = mapped_position, y = y_label, fill = value), width = tile_width, height = tile_height, colour="grey") +
  scale_fill_gradient(low = "white", high = COUNTERFACTUAL_COLOR, limits=range(data_final$value)) +
  scale_x_continuous(breaks = seq(0, length(relevant_position) - 1,1), labels = relevant_position) +
  scale_y_discrete(limits = unique(data_final$y_label)) +
  labs(fill = "Attention\nScore:") +
  theme(
    axis.text.x = element_text(size=60, angle = 45, hjust = 1),
    axis.text.y = element_text(size=60, angle = 0),
   # axis.text.x = element_text(size=40, angle = 45, hjust = 1),
   # axis.text.y = element_text(size=40, angle = 0),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    legend.text = element_text(size = 45),
    legend.title = element_text(size = 50),
    legend.position = "right",
    legend.key.size = unit(1.3, "cm"),
  )
heatmap_plot
ggsave(sprintf("paper_plots/%s_%s_heads_pattern/head_pattern_layer.pdf", model, experiment), heatmap_plot, width = 53, height = 38, units = "cm", create.dir = TRUE)





### Full position
#### FUnctions

create_heatmap <- function(data, x, y, fill,title, color) {
    p <- create_heatmap_base(data, x, y, fill) +
      scale_fill_gradient2(low = LOW_COLOR, mid = "white", high = color, midpoint = 0, limits=c(0,0.45)) +
      theme_minimal() +
      #addforce to have all the labels
      scale_y_discrete(breaks = seq(0, length(relevant_position)-1,1), labels = relevant_position) +
      scale_x_discrete(breaks = seq(0, length(relevant_position)-1,1), labels = relevant_position) +
      labs(fill = "Attention\nscore:", title=title) +
      theme(
        axis.text.x = element_text(size=AXIS_TEXT_SIZE-10, angle = 45, hjust = 1),
        axis.text.y = element_text(size=AXIS_TEXT_SIZE-10),
        title = element_text(size = AXIS_TEXT_SIZE-10),
        #remove background grid
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        legend.text = element_text(size = 30),
        legend.title = element_text(size = 40),
        #remove the legend\
        legend.position = "right",
        legend.key.size = unit(1.3, "cm"),
        # move the y ticks to the right
      )
    return(p)
}
  plot_pattern <- function(l, h, data,color){
    selected_layer <- l
    selected_head <- h
    data_head <- data %>% filter(layer == selected_layer & head == selected_head)
    max_source_position <- max(as.numeric(data_head$source_position))
    max_dest_position <- max(as.numeric(data_head$dest_position))
    data_head$source_position <- factor(data_head$source_position, levels = c(0:max_source_position))
    data_head$dest_position <- factor(data_head$dest_position, levels = c(0:max_dest_position))
    #filter just the relevant positions for both source_position and dest_position
    data_head <- data_head %>% filter(dest_position == 1 | dest_position==4 | dest_position == 5 | dest_position== 6 | dest_position==8 |  dest_position == 11 | dest_position== 12)
    data_head <- data_head %>% filter(source_position == 1 | source_position==4 | source_position == 5 | source_position== 6 | source_position==8 |  source_position == 11 | source_position== 12)
    #remap the position
    unique_positions <- unique(data_head$dest_position)
    position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
    # Apply the mapping to create a new column
    data_head$dest_mapped <- unname(position_mapping[as.character(data_head$dest_position)])
    data_head$source_mapped <- unname(position_mapping[as.character(data_head$source_position)])
    # order the position
    data_head$dest_mapped <- factor(data_head$dest_mapped, levels = c(0,1,2,3,4,5,6))
    data_head$source_mapped <- factor(data_head$source_mapped, levels = c(6,5,4,3,2,1,0))
    
    #reorder the source_position and dest_position contrary to the order of the factor
    
    #data_head$source_mapped <- factor(data_head$source_mapped, levels = rev(levels(data_head$source_mapped)))

    p <- create_heatmap(data_head, "dest_mapped", "source_mapped", "value", paste("Layer", l, "Head", h), color)
    p
    return(p)
  }

#### Process Data

pattern_df <- data.frame(layer = layer_pattern, head = head_pattern)
#select the head that are in pattern_df (the whole tuple layer, head)
data <- merge(data, pattern_df, by = c("layer", "head"))
#filter just the relevant positions for both source_position and dest_position
data <- data %>% filter(dest_position == 1 | dest_position==4 | dest_position == 5 | dest_position== 6 | dest_position==8 |  dest_position == 11 | dest_position== 12)
data <- data %>% filter(source_position == 1 | source_position==4 | source_position == 5 | source_position== 6 | source_position==8 |  source_position == 11 | source_position== 12)
#mapped position


unique_positions <- unique(data$dest_position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data$dest_mapped <- unname(position_mapping[as.character(data$dest_position)])
data$source_mapped <- unname(position_mapping[as.character(data$source_position)])
# order the position 
data$dest_mapped <- factor(data$dest_mapped, levels = unique(data$dest_mapped))
data$source_mapped <- factor(data$source_mapped, levels = unique(data$source_mapped))

#select a specific head
data_head <- data %>% filter(layer == 11 & head == 10)

library(ggplot2)

# Reorder factors to have origin at the top and labels for future use
source_mapped_levels <- rev(levels(data_head$source_mapped))
dest_mapped_levels <- levels(data_head$dest_mapped)
scale_y_discrete(limits = source_mapped_levels)

#### Plot

# plot <- NULL
# for (i in c(1:length(head_pattern))) {
#   head <- head_pattern[i]
#   layer <- layer_pattern[i]
#   if (layer %in% factual_heads_layer & head %in% factual_heads_head){
#     color <-FACTUAL_COLOR
#   }else{
#     color <-  COUNTERFACTUAL_COLOR
#   }
#    plot <- plot + plot_pattern(layer,head, data, color)
# }
# plot + plot_layout(ncol=2, nrow=3)

library(patchwork)

# plots <- list()
# for (i in c(1:length(head_pattern))) {
#   head <- head_pattern[i]
#   layer <- layer_pattern[i]
#   if (layer %in% factual_heads_layer & head %in% factual_heads_head){
#     print(layer)
#     print(head)
#     color <- FACTUAL_COLOR
#   }else{
#     color <-  COUNTERFACTUAL_COLOR
#   }
#   print(color)
#   plots <- c(plots, list(plot_pattern(layer, head, data, color)))
# }

plots <- list()
for (i in c(1:length(head_pattern))) {
  head <- head_pattern[i]
  layer <- layer_pattern[i]
  if ((layer==11 & head==10) || (layer==10 & head==7)){
    print(layer)
    print(head)
    color <- FACTUAL_COLOR
  }else{
    color <-  COUNTERFACTUAL_COLOR
  }
  print(color)
  plots <- c(plots, list(plot_pattern(layer, head, data, color)))
}

plot <- wrap_plots(plots, ncol = 2, nrow = 3)
ggsave(sprintf("paper_plots/%s_%s_heads_pattern/full_pattern.pdf", model, experiment), plot, width = 80, height = 100, units = "cm")



## Boosting Heads

data_long <- data.frame(
  model = c("GPT2","GPT2", "Pythia-6.9b",  "Pythia-6.9b"),
  Type = c("Baseline", "Multiplied Attention\nAltered","Baseline", "Multiplied Attention\nAltered" ),
  Percentage = c(4.13,50.29,30.32,49.46) # source?
)
p <- ggplot(data_long, aes(x = model, y = Percentage, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge", color="black", size=1.4) +
  #geom_text(aes(label = label), vjust = -0.5, position = position_dodge(width = 0.9), na.rm = TRUE, size=14) +
  scale_fill_manual(values = c("#ff6361","#003f5c"), labels= c("Baseline", expression(alpha == 5))) +
  labs(x = "",
       y = "% factual answers") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 70, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 70),
    axis.title.y = element_text(size = 75),
    legend.text = element_text(size = 65),
    legend.title = element_blank(),
    legend.position = "bottom",
    legend.text.align = 0.5, # Center align text relative to the keys
    legend.spacing.x = unit(1.5, "cm")
  ) +
  guides(fill = guide_legend(ncol = 2.5)) # Adjusting the legend
p
ggsave(sprintf("paper_plots/%s_%s_heads_pattern/multiplied_pattern.pdf", model, experiment), p, width = 60, height = 40, units = "cm")