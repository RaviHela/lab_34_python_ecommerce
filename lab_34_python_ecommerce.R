# BUSINESS SCIENCE UNIVERSITY
# LEARNING LABS
# LAB 34: eCommerce Classifying Customers ----

# LIBRARIES ----

library(tidyverse)
library(timetk)
library(lubridate)
library(reticulate)

# PYTHON ENVIRONMENT ----

reticulate::use_condaenv("py3.8", required = TRUE)
reticulate::py_config()

# DATA ----

ecommerce_raw_tbl <- read_csv("data/ecommerce_data.csv")
ecommerce_raw_tbl

ecommerce_tbl <- ecommerce_raw_tbl %>%
    mutate(InvoiceDate = mdy_hms(InvoiceDate)) %>%
    select(contains("ID"), contains("Invoice"), everything()) %>%
    mutate(CustomerID = as.character(CustomerID)) %>%
    mutate(PriceExt = Quantity * UnitPrice)

ecommerce_tbl

ecommerce_tbl %>%
    summarise_by_time(InvoiceDate, .by = "day", revenue = sum(PriceExt)) %>%
    plot_time_series(InvoiceDate, revenue, .smooth_period = "3 months")
