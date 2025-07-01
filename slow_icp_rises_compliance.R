# Load necessary libraries
library('lme4')
library('lmerTest')
library('readxl')
library('sjPlot')
library('dplyr')
library('writexl')

# FOLDER PATH
folder = "N:/tiger/baptiste.balanca/Neuro_rea_monitorage/figures/slow_icp_rises_figs/res_matrix/"

### LOAD DATA
path <- paste0(folder,"compliance.xlsx")
data <- read_excel(path)

# Ensure 'win_label' is a factor
data$win_label <- factor(data$win_label, 
                         levels = c('baseline_before', 'rise_1', 'rise_2', 'rise_3', 'rise_4','rise_5', 
                                    'decay_1', 'decay_2', 'decay_3', 'decay_4', 'decay_5','baseline_after'), 
                         ordered = FALSE)

# Define labels & dependent variables
win_labels <- levels(data$win_label)  # Automatically gets all labels
dependent_vars <- c('median_icp_mmHg','median_abp_mmHg','median_cpp_mmHg',
                    'heart_in_icp_spectrum','resp_in_icp_spectrum','ratio_heart_resp_in_icp_spectrum',
                    'icp_pulse_amplitude_mmHg','icp_resp_modulated','icp_pulse_resp_modulated',
                    'abp_pulse_amplitude_mmHg','abp_resp_modulated','abp_pulse_resp_modulated',
                    'P2P1_ratio','PSI','PRx',
                    'RAQ_2','RAQ_ABP','heart_rate_bpm',
                    'resp_rate_cpm')

# Initialize results dataframe
results <- data.frame(matrix(ncol = length(win_labels), nrow = length(dependent_vars)))
colnames(results) <- win_labels
rownames(results) <- dependent_vars

# Function to format p-values with significance stars
get_significance <- function(p_value) {
  if (p_value <= 0.001) {
    return("***")
  } else if (p_value <= 0.01) {
    return("**")
  } else if (p_value <= 0.05) {
    return("*")
  } else {
    return("ns")  # Not significant
  }
}

# Loop over dependent variables
for (dep_var in dependent_vars) {
  # Fit the mixed model
  formula <- as.formula(paste(dep_var, "~ win_label + Norépinéphrine + Midazolam + Propofol + Kétamine + Thiopental + Eskétamine + Rémifentanil + Sufentanil + Morphine +  (1|n_event_sub)"))  # Random effect: 'Subject'
  model <- tryCatch(lmer(formula, data = data), error = function(e) NULL)
  
  # Skip if model fails
  if (is.null(model)) next
  
  # Extract coefficients & confidence intervals
  coefs <- summary(model)$coefficients
  conf_int <- confint(model, method = "Wald")  # Compute confidence intervals
  
  for (win in win_labels) {
    contrast_name <- paste0("win_label", win)
    
    if (contrast_name %in% rownames(coefs)) {
      estimate <- round(coefs[contrast_name, "Estimate"], 3)
      p_value <- coefs[contrast_name, "Pr(>|t|)"]
      
      # Handle very small p-values
      p_value_formatted <- ifelse(p_value < 0.001, "< 0.001", round(p_value, 3))
      
      # Get significance stars
      significance <- get_significance(p_value)
      
      # Confidence intervals
      ci_lower <- round(conf_int[contrast_name, 1], 3)
      ci_upper <- round(conf_int[contrast_name, 2], 3)
      
      # Store in table with significance stars
      results[dep_var, win] <- paste0("β=", estimate, 
                                      " [", ci_lower, ", ", ci_upper, "], p=", p_value_formatted, 
                                      ", ", significance)
    } else if (win == "baseline_before") {
      # Compute median for baseline_before
      baseline_data <- data[data$win_label == "baseline_before", dep_var]
      baseline_data <- as.numeric(unlist(baseline_data))
      median_baseline <- round(median(baseline_data, na.rm = TRUE), 3)
      
      # Store Reference (β=0) + median info
      results[dep_var, win] <- paste0("Reference (β=0), Median=", median_baseline)
    } else {
      results[dep_var, win] <- "NA"
    }
  }
}

row_name_dict <- c(
  "median_icp_mmHg" = "ICP (mmHg)",
  "heart_in_icp_spectrum" = "Heart in ICP Spectrum (mmHg)",
  "icp_pulse_amplitude_mmHg" = "ICP Pulse Amplitude (mmHg)",
  "resp_in_icp_spectrum" = "Respiration in ICP Spectrum (mmHg)",
  "icp_resp_modulated" = "Respiratory Modulation of ICP (mmHg)",
  "ratio_heart_resp_in_icp_spectrum" = "Heart / Resp Spectral Ratio",
  "icp_pulse_resp_modulated" = "Respiratory Modulation of ICP Pulse Amplitude (mmHg)",
  "median_abp_mmHg" = "ABP (mmHg)",
  "median_cpp_mmHg" = "CPP (mmHg)",
  "abp_pulse_amplitude_mmHg" = "ABP Pulse Amplitude (mmHg)",
  "abp_pulse_resp_modulated" = "Respiratory Modulation of ABP Pulse Amplitude (mmHg)",
  "abp_resp_modulated" = "Respiratory Modulation of ABP (mmHg)",
  "RAQ_2" = "RAQ_2",
  "RAQ_ABP" = "RAQ_ABP",
  "P2P1_ratio" = "P2P1 Ratio",
  "PSI" = "PSI",
  "PRx" = "PRx",
  "heart_rate_bpm" = "Heart Rate (bpm)",
  "resp_rate_cpm" = "Respiratory Rate (cpm)"
)

# Replace row names using the dictionary
rownames(results) <- row_name_dict[rownames(results)]

# Create a dictionary for window labels to replace in columns
col_name_dict <- c(
  'baseline_before' = "Baseline Before",
  'rise_1' = "Rise 1",
  'rise_2' = "Rise 2",
  'rise_3' = "Rise 3",
  'rise_4' = "Rise 4",
  'rise_5' = "Rise 5",
  'decay_1' = "Decay 1",
  'decay_2' = "Decay 2",
  'decay_3' = "Decay 3",
  'decay_4' = "Decay 4",
  'decay_5' = "Decay 5",
  'baseline_after' = "Baseline After"
)

# Replace column names using the dictionary
colnames(results) <- col_name_dict[colnames(results)]

# Print final results
print(results)

# Add row names as a column
results_with_rownames <- cbind(RowName = rownames(results), results)
colnames(results_with_rownames)[1] <- "Metric / Window Label"
write_xlsx(results_with_rownames, paste0(folder,"results_compliance_lmm.xlsx"))

