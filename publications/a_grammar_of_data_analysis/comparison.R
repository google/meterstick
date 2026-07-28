# ==============================================================================
# comparison.R
#
# Description:
#   This script replicates the data analysis examples from the paper
#   "A Grammar of Data Analysis" using R and the tidyverse (dplyr, readr, tidyr).
#   It serves as a comparison to the Python/Meterstick implementations.
#
# Examples covered:
#   1. Baseball Data Analysis (pitch distribution)
#   2. Churn Rate (A/B test comparison)
#   3. Difference-in-Differences (minimum wage study)
#
# Data Sources:
#   - Reads from a local 'datasets/' directory by default.
#   - Can be configured to read directly from GitHub by uncommenting the
#     relevant lines in each example.
# ==============================================================================

library(dplyr)
library(readr)
library(tidyr)

# ==============================================================================
# Example 1: Baseball Data Analysis
# ==============================================================================
cat("\n--- Running Example 1: Baseball Data Analysis ---\n")

# Read directly from GitHub:
# df_baseball <- read_csv("https://raw.githubusercontent.com/google/meterstick/refs/heads/master/publications/a_grammar_of_data_analysis/datasets/baseball_pitches_to_SEA_batters.csv")
# player_names <- read_csv("https://raw.githubusercontent.com/google/meterstick/refs/heads/master/publications/a_grammar_of_data_analysis/datasets/baseball_player_names.csv")

# Or read from a local directory:
df_baseball <- read_csv("datasets/baseball_pitches_to_SEA_batters.csv")
player_names <- read_csv("datasets/baseball_player_names.csv")

player_names <- player_names |>
  mutate(Name = paste(first_name, last_name, sep = " "))

df_baseball <- df_baseball |>
  left_join(player_names |> select(id, Name), by = c("pitcher_id" = "id"))
df_baseball <- df_baseball |>
  rename(Name_pitcher = Name) |>
  left_join(player_names |> select(id, Name), by = c("batter_id" = "id")) |>
  rename(Name_batter = Name)

baseball_res <- df_baseball |>
  group_by(Name_batter, Name_pitcher, pitch_type) |>
  count() %>%
  group_by(Name_pitcher, Name_batter) %>%
  mutate(n / sum(n))

# Show a sample to compare with Python output
sample_baseball <- baseball_res |>
  filter(Name_batter == "Dee Gordon", Name_pitcher == "A.J. Cole")
print(sample_baseball)


# ==============================================================================
# Example 2: Churn Rate (Online Retailer)
# ==============================================================================
cat("\n--- Running Example 2: Churn Rate ---\n")

# Read directly from GitHub:
# df_churn <- read_delim("https://raw.githubusercontent.com/google/meterstick/refs/heads/master/publications/a_grammar_of_data_analysis/datasets/churn_rate.csv", delim = ";")

# Or read from a local directory:
df_churn <- read_delim("datasets/churn_rate.csv", delim = ";")

df_churn <- df_churn |> mutate(lost = as.logical(lost))

df_by_expt <- df_churn |>
  group_by(region, experiment) |>
  summarize(churn = sum(lost, na.rm = TRUE) / n())

df_treated <- df_by_expt |> filter(experiment != "control")
df_control <- df_by_expt |> filter(experiment == "control")

churn_change <- df_treated |>
  inner_join(df_control, by = "region", suffix = c("_treated", "_control")) |>
  mutate(churn_diff = 100 * (churn_treated / churn_control - 1)) |>
  select(region, experiment = experiment_treated, churn_diff)

print(churn_change)


# ==============================================================================
# Example 3: Difference-in-Differences (Minimum Wage)
# ==============================================================================
cat("\n--- Running Example 3: Difference-in-Differences ---\n")

# Read directly from GitHub:
# df_minwage <- read_delim("https://raw.githubusercontent.com/google/meterstick/refs/heads/master/publications/a_grammar_of_data_analysis/datasets/minimum_wage.csv", delim = ";")

# Or read from a local directory:
df_minwage <- read_delim("datasets/minimum_wage.csv", delim = ";")

df_minwage <- df_minwage |>
  mutate(
    STATE_NAME = ifelse(STATE == 1, "NJ", "PA"),
    CHAIN = case_when(
      CHAIN == 1 ~ "Burger King",
      CHAIN == 2 ~ "KFC",
      CHAIN == 3 ~ "Roy Rogers",
      CHAIN == 4 ~ "Wendy's",
      TRUE ~ as.character(CHAIN)
    ),
    EMPTOT = EMPPT * 0.5 + EMPFT + NMGRS,
    EMPTOT2 = EMPPT2 * 0.5 + EMPFT2 + NMGRS2
  )

df_long <- df_minwage |>
  pivot_longer(
    cols = c(EMPTOT, EMPTOT2),
    names_to = "PERIOD",
    values_to = "EMP"
  ) |>
  mutate(
    PERIOD = recode(PERIOD, EMPTOT = "Before", EMPTOT2 = "After")
  )

did_res <- df_long |>
  group_by(STATE_NAME, PERIOD) |>
  summarize(EMP_RATE = mean(EMP, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = STATE_NAME, values_from = EMP_RATE) |>
  mutate(DIFF = NJ - PA) |>
  select(PERIOD, DIFF) |>
  pivot_wider(names_from = PERIOD, values_from = DIFF) |>
  mutate(DIFF_OF_DIFFS = After - Before)

print(did_res)
