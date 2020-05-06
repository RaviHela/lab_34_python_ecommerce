# LEARNING LAB 33: HR EMPLOYEE CLUSTERING WITH PYTHON
# BONUS #1 - EMPLOYEE ATTRITION EXPLORER

# LIBRARIES ----
library(shiny)
library(shinythemes)
library(DT)

library(tidyverse)
library(tidyquant)

# Preprocessing
library(recipes)

# Python
library(reticulate)

# PYTHON SETUP ----

# Replace this with your conda environment containking sklearn, pandas, & numpy
use_condaenv("py3.8", required = TRUE)
source_python("py/logistic_reg.py")

# DATA SETUP ----
ecommerce_raw_tbl  <- read_csv("data/ecommerce_data.csv")
invoice_selections <- ecommerce_raw_tbl %>%
    filter(!InvoiceNo %>% str_detect("^C")) %>%
    distinct(InvoiceNo) %>% 
    slice(1:30) %>%
    pull()

sample_data <- ecommerce_raw_tbl %>% filter(InvoiceNo %in% invoice_selections)

customer_order_history_tbl <- read_rds("preprocessing/customer_habits_joined_tbl.rds") %>%
    select(CustomerID, count)

product_clusters_tbl <- read_rds("preprocessing/product_clusters_tbl.rds") 

recipe_spec <- read_rds("preprocessing/recipe_spec_customer_prediction.rds")

cluster_morphology_tbl <- read_rds("preprocessing/cluster_morphology_tbl.rds")

cluster_morphology_ggplot <- read_rds("preprocessing/cluster_morphology_ggplot.rds")


# UI ----
ui <- navbarPage(
    title = "E-Commerce App",
    collapsible = TRUE,
    position    = "static-top", 
    inverse     = TRUE, 
    theme       = shinytheme("paper"),
    
    tabPanel(
        title = "Select",
        sidebarLayout(
            sidebarPanel = sidebarPanel(
                width = 3,
                h3("Market Basket"),
                shiny::selectInput(
                    inputId  = "invoice_selection", 
                    label    = "Analyze an Invoice",
                    choices  = invoice_selections,
                    selected = invoice_selections[1]
                        
                ),
                shiny::actionButton(inputId = "submit", "Submit", class = "btn-primary"),
                hr(),
                uiOutput("text")
                
            ),
            mainPanel = mainPanel(
                width = 9,
                div(
                    class = "row",
                    div(
                        class = "col-sm-12 panel",
                        div(class = "panel-heading", h5("Customer Segment Morphology")),
                        div(
                            class = "panel-body",
                            plotOutput("ggplot", height = "275px"),
                            # verbatimTextOutput(outputId = "print")
                        )
                    )
                ),
                div(
                    class = "row",
                    div(
                        class = "col-sm-12 panel",
                        div(class = "panel-heading", h5("Recommended Products")),
                        div(
                            class = "panel-body",
                            dataTableOutput("recommendation")
                        )
                    )
                ),
                div(
                    class = "row",
                    div(
                        class = "col-sm-12 panel",
                        div(class = "panel-heading", h5("Customer's Basket")),
                        div(
                            class = "panel-body",
                            dataTableOutput("basket"),
                        )
                    )
                )
            )
        )
    )
)

# SERVER ---- 
server <- function(session, input, output) {
    rv <- reactiveValues()
    
    observeEvent(input$submit, {
        
        rv$invoice_tbl <- ecommerce_raw_tbl %>%
            filter(InvoiceNo %in% input$invoice_selection) %>%
            mutate(PriceExt = UnitPrice * Quantity) %>%
            left_join(product_clusters_tbl %>% select(StockCode, cluster))
        
        # Count (Invoice History)
        customer_id <- rv$invoice_tbl %>%
            distinct(CustomerID) %>%
            pull()
        
        rv$count <- customer_order_history_tbl %>%
            filter(CustomerID == customer_id) %>%
            slice(1) %>%
            pull(count)
        
        
        # Mean (Total Invoice Value)
        rv$mean <- rv$invoice_tbl %>%
            pull(PriceExt) %>%
            sum(na.rm = TRUE)
        
        # Category Proportions (Product Clusters)
        empty_tbl <- c(str_c("cat_", 0:5), "cat_NA") %>% 
            purrr::map_dfc(setNames, object = list(numeric()))
        
        rv$product_props_by_cat <- rv$invoice_tbl %>%
            select(CustomerID, PriceExt, cluster) %>%
            group_by(CustomerID, cluster) %>%
            summarize(PriceExt = SUM(PriceExt)) %>%
            mutate(prop = PriceExt / SUM(PriceExt)) %>%
            ungroup() %>%
            select(-PriceExt) %>%
            pivot_wider(names_from   = cluster, 
                        values_from  = prop, 
                        values_fill  = list(prop = 0), 
                        names_prefix = "cat_") %>%
            select(-CustomerID) %>%
            bind_rows(empty_tbl)
        
        # Fill missing categories with zero    
        rv$product_props_by_cat[is.na(rv$product_props_by_cat)] <- 0
        
        # Make preprocessing table
        rv$data_bind <- tibble(
            count = rv$count,
            mean  = rv$mean
        ) %>%
            bind_cols(rv$product_props_by_cat) 
        
        # Apply recipe & reorder to match column structure for previously trainedLogistic Regression Model
        rv$data_prep <- bake(recipe_spec, new_data = rv$data_bind) %>%
            select(count, mean, cat_2, cat_0, cat_1, cat_3, cat_NA, cat_4)
        
        
        # Make prediction
        rv$prediction <- lr_predict(as.matrix(rv$data_prep)) %>% as.character()
        
        # Product Recommendation
        
        product_cat <- cluster_morphology_tbl %>%
            mutate(cluster = as.character(cluster) %>% as.numeric()) %>%
            filter(cluster == rv$prediction) %>%
            select(cluster, contains("cat_")) %>%
            pivot_longer(-cluster) %>%
            arrange(desc(value)) %>%
            slice(1) %>%
            separate(name, into = c("unnecessary", "product_cat")) %>%
            mutate(product_cat = as.numeric(product_cat)) %>%
            pull(product_cat) %>%
            as.character()
        
        if (is.na(product_cat)) product_cat <- "5"
        
        rv$recommendation_tbl <- ecommerce_raw_tbl %>%
            select(StockCode, Quantity) %>%
            group_by(StockCode) %>%
            summarize(Quantity = SUM(Quantity)) %>%
            ungroup() %>%
            left_join(product_clusters_tbl) %>%
            mutate(product_category = as.character(cluster)) %>%
            mutate(product_category = ifelse(is.na(product_category), "5", product_category)) %>%
            filter(product_category == product_cat) %>%
            arrange(desc(Quantity)) %>%
            slice(1:3) %>%
            select(StockCode, mode_description, median_unit_price, product_category)
        
     
        
    }, ignoreNULL = FALSE)
    
    # Debugging
    output$print <- renderPrint({
        list(
            invoice_tbl    = rv$invoice_tbl,
            # data_prep    = rv$data_prep,
            prediction     = rv$prediction,
            recommendation = rv$recommendation_tbl
        )
    })
    
    output$basket <- renderDataTable({
        rv$invoice_tbl %>%
            rename(`Product Category` = cluster) %>%
            select(-InvoiceNo) %>%
            datatable()
    })
    
    output$recommendation <- renderDataTable({
        rv$recommendation_tbl %>%
            datatable()
    })
    
    output$text <- renderUI({
        div(
            h5("Customer Segment Prediction: ", span(rv$prediction, class="label label-primary"))
        )
    })
    
    # GGPLOT ----
    output$ggplot <- renderPlot({
        cluster_morphology_ggplot +
            facet_wrap(~ cluster, nrow = 1) +
            theme(
                strip.text = element_text(color = "white"),
                plot.margin = unit(c(0,0,0,0),"mm"), 
                text = element_text(size = 16)
            ) +
            labs(title = "")
    })
    
    
}

# Run the application 
shinyApp(ui = ui, server = server)