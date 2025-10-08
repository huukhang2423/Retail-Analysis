import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import (
    col, year, month, dayofmonth, hour, count, 
    sum as spark_sum, desc, avg, to_timestamp, 
    countDistinct, date_format, lag, lit, when,
    from_unixtime, percentile_approx, datediff,
    split,collect_list,collect_set, 
)
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType, DateType

# Setup JAVA_HOME for PySpark
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-20'
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-class-path . pyspark-shell'

# Set page title
st.set_page_config(
    page_title="Online Retail Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Create page title with improved design
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>📊 Online Retail Dashboard </h1>
""", unsafe_allow_html=True)

# File path 
parquet_file_path = r"C:\Users\Admin\Desktop\big_data_project\online_retail_data\online_retail_spark.parquet"

# Initialize Spark session 
@st.cache_resource(show_spinner=False)
def get_spark():
    # Create a progress bar 
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Update progress
        progress_text.text("Initializing Spark session...")
        progress_bar.progress(20)
        
        # Initialize Spark session 
        spark = SparkSession.builder \
            .appName("OnlineRetailBigData") \
            .config("spark.executor.memory", "6g") \
            .config("spark.driver.memory", "10g") \
            .config("spark.sql.session.timeZone", "UTC") \
            .config("spark.driver.allowMultipleContexts", "true") \
            .master("local[*]") \
            .getOrCreate()
        
        progress_text.text("Spark session initialized successfully!")
        progress_bar.progress(100)
        import time
        time.sleep(1)
        
    # Clear the progress elements 
    progress_placeholder.empty()
    
    return spark

# Ensure SparkContext is active
def ensure_spark_context():
    try:
        spark = get_spark()
        # Test if SparkContext is active
        _ = spark.sparkContext.parallelize([1]).count()
        return spark
    except Exception as e:
        st.error(f"Error initializing or testing Spark: {e}")
        # Try to create a new session if possible
        try:
            spark = SparkSession.builder \
                .appName("OnlineRetailBigData-Retry") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "8g") \
                .master("local[*]") \
                .getOrCreate()
            return spark
        except Exception as retry_error:
            st.error(f"Failed to initialize Spark after retry: {retry_error}")
            return None

# Function to read data
@st.cache_resource(show_spinner=False)
def load_data(_spark):
    try:
        # Create a progress bar for data loading
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Update progress
            progress_text.text("Reading data with Spark...")
            progress_bar.progress(20)
            
            if not os.path.exists(parquet_file_path):
                progress_placeholder.empty()
                st.error(f"File not found: {parquet_file_path}")
                return None
                
            spark_df = _spark.read.parquet(parquet_file_path)
            
            # Update progress
            progress_text.text("Analyzing data schema...")
            progress_bar.progress(40)
            
            # Debug: Print schema but don't show in UI
            spark_df.printSchema()
            
            # Update progress
            progress_text.text("Normalizing data...")
            progress_bar.progress(60)
            
            # StockCode validation
            spark_df = spark_df.withColumn(
                "StockCode", 
                functions.trim(functions.upper(col("StockCode")))
            )
            
            # Update progress
            progress_text.text("Optimizing data for analysis...")
            progress_bar.progress(80)
            
            # Cache DataFrame to improve performance
            spark_df.cache()
            
            # Count rows to trigger cache
            row_count = spark_df.count()
            
            # Update progress
            progress_text.text(f"Data loaded successfully with {row_count:,} rows")
            progress_bar.progress(100)
            
            # Wait a moment before clearing
            import time
            time.sleep(1.5)
            
        # Clear the progress elements after completion
        progress_placeholder.empty()
        
        return spark_df
        
    except Exception as e:
        if 'progress_placeholder' in locals():
            progress_placeholder.empty()
        st.error(f"Error during data processing with Spark: {e}")
        return None

# Function to filter data using Spark
def filter_spark_data(df, countries=None, start_date=None, end_date=None):
    try:
        filtered_df = df
        
        # Filter by Country if selected
        if countries and len(countries) > 0:
            filtered_df = filtered_df.filter(col("Country").isin(countries))
        
        # Filter by time range
        if start_date and end_date:
            # Convert to timestamps safely
            start_timestamp = pd.Timestamp(start_date).timestamp()
            end_timestamp = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).timestamp()
            
            filtered_df = filtered_df.filter(
                (col("InvoiceDate") >= from_unixtime(lit(start_timestamp))) & 
                (col("InvoiceDate") <= from_unixtime(lit(end_timestamp)))
            )
        
        return filtered_df
    except Exception as e:
        st.error(f"Error during filtering: {e}")
        # Return the original dataframe if filtering fails
        return df

# Main execution starts here
with st.spinner("Loading big data application..."):
    # Get Spark session with error handling
    spark = ensure_spark_context()
    
    if not spark:
        st.error("Could not initialize Spark. Please restart the application.")
        st.stop()
    
    # Load data with progress bar
    spark_df = load_data(spark)
    
    if spark_df is None:
        st.error("Unable to load data. Please check the file and path.")
        st.stop()
    else:
        # Display data information
        st.success(f"Data loaded successfully with {spark_df.count():,} rows")

# Create filters with improved design
st.sidebar.markdown("""
    <h2 style='text-align: center; color: #2E86C1;'>🔍 Filters</h2>
    <hr>
""", unsafe_allow_html=True)

# 1. Country filter - use Spark to get unique countries
try:
    countries = [row.Country for row in spark_df.select("Country").distinct().orderBy("Country").collect()]
    selected_countries = st.sidebar.multiselect(
        "Select countries:",
        options=countries,
        default=["United Kingdom"]  # Default to United Kingdom as it has the most data
    )
except Exception as e:
    st.sidebar.error(f"Error loading countries: {e}")
    selected_countries = []

# 2. Time filter - use Spark to get min/max dates
try:
    date_range = spark_df.agg(
        functions.min("InvoiceDate").alias("min_date"),
        functions.max("InvoiceDate").alias("max_date")
    ).collect()[0]

    min_date = date_range.min_date.date()
    max_date = date_range.max_date.date()

    # Create 2 date_input to select time range
    start_date = st.sidebar.date_input("From date:", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("To date:", max_date, min_value=min_date, max_value=max_date)
except Exception as e:
    st.sidebar.error(f"Error loading date range: {e}")
    # Set default dates if there's an error
    today = datetime.now().date()
    start_date = today - timedelta(days=30)
    end_date = today

# Apply filters with error handling
try:
    # Show a spinner while filtering
    with st.spinner("Applying filters..."):
        filtered_spark_df = filter_spark_data(spark_df, selected_countries, start_date, end_date)

        # Check if any data remains after filtering
        row_count = filtered_spark_df.count()
        
    if row_count == 0:
        st.warning("No data matches the selected filters. Please change the filters.")
        st.stop()
    else:
        st.info(f"Filtered data contains {row_count:,} rows")
except Exception as e:
    st.error(f"Error applying filters: {e}")
    filtered_spark_df = spark_df  # Use unfiltered data as fallback
    st.info(f"Using unfiltered data due to filter error: {spark_df.count():,} rows")

# Create dashboard with 4 tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🌍 Country Analysis", "⏱️ Time Analysis", "🔗 Associate Analysis"])
# Define colors and themes 
color_theme = px.colors.qualitative.Plotly
main_color = "#2E86C1"  # Main color
secondary_color = "#85C1E9"  # Secondary color

with tab1:
    st.markdown("""
        <h2 style='text-align: center; color: #2E86C1;'>📊 Sales Data Overview</h2>
    """, unsafe_allow_html=True)
    
    # Calculate KPIs with Spark
    try:
        kpi_data = filtered_spark_df.agg(
            spark_sum("TotalValue").alias("total_revenue"),
            countDistinct("InvoiceNo").alias("total_invoices"),
            countDistinct("CustomerID").alias("total_customers")
        ).collect()[0]
        
        # Create layout with 4 columns for main KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        # KPI 1: Total revenue
        total_revenue = kpi_data.total_revenue
        col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
        
        # KPI 2: Number of orders
        total_invoices = kpi_data.total_invoices
        col2.metric("🛍️ Total Orders", f"{total_invoices:,}")
        
        # KPI 3: Number of customers
        total_customers = kpi_data.total_customers
        col3.metric("👥 Total Customers", f"{total_customers:,}")
        
        # KPI 4: Average order value
        avg_order_value = total_revenue / total_invoices if total_invoices > 0 else 0
        col4.metric("💵 Avg Order Value", f"${avg_order_value:.2f}")
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
    
    st.markdown("---")
    
    # Display monthly revenue chart with Plotly
    st.subheader("📅 Monthly Revenue")
    
    try:
        # Calculate total revenue by month and year with Spark
        monthly_revenue_spark = filtered_spark_df.groupBy("Year", "Month") \
            .agg(spark_sum("TotalValue").alias("Revenue")) \
            .orderBy("Year", "Month")
        
        # Convert to Pandas for visualization
        monthly_revenue = monthly_revenue_spark.toPandas()
        monthly_revenue['Date'] = monthly_revenue.apply(lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1)
        
        # Create chart with Plotly
        fig = px.bar(
            monthly_revenue, 
            x='Date', 
            y='Revenue',
            labels={'Date': 'Month', 'Revenue': 'Revenue'},
            title='Monthly Revenue',
            color_discrete_sequence=[main_color],
            template='plotly_white'
        )
        
        # Improve display
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Revenue',
            hovermode='x unified',
            hoverlabel=dict(bgcolor="white", font_size=12),
            height=500
        )
        
        # Display formatted amounts on hover
        fig.update_traces(
            hovertemplate='Month: %{x}<br>Revenue: $%{y:.2f}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating monthly revenue chart: {e}")
    
    # Top 10 best-selling products with Plotly
    st.subheader("🔝 Top 10 Best-Selling Products")
    
    try:
        # Calculate total quantity sold by product using Spark
        top_products_spark = filtered_spark_df.filter(col("Description").isNotNull()) \
            .groupBy("Description") \
            .agg(spark_sum("Quantity").alias("Quantity")) \
            .orderBy(desc("Quantity")) \
            .limit(10)
        
        # Check if there are enough products
        product_count = top_products_spark.count()
        if product_count < 10:
            st.info(f"Only {product_count} different products in the filtered data")
            
        # Convert to Pandas and sort for display
        top_products_df = top_products_spark.toPandas()
        top_products_df = top_products_df.sort_values('Quantity', ascending=True)  # For bottom-up display
        
        if not top_products_df.empty:
            # Create chart with Plotly
            fig = px.bar(
                top_products_df, 
                y='Description', 
                x='Quantity',
                orientation='h',
                labels={'Description': 'Product', 'Quantity': 'Quantity Sold'},
                title=f'Top {product_count} Best-Selling Products',
                color_discrete_sequence=[secondary_color],
                template='plotly_white'
            )
            
            # Improve display
            fig.update_layout(
                xaxis_title='Quantity Sold',
                yaxis_title='Product',
                hovermode='y unified',
                hoverlabel=dict(bgcolor="white", font_size=12),
                height=600
            )
            
            # Limit product name length
            fig.update_yaxes(tickfont=dict(size=10))
            
            # Add quantity to horizontal bars
            fig.update_traces(
                texttemplate='%{x}',
                textposition='outside',
                hovertemplate='Product: %{y}<br>Quantity Sold: %{x}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough product data to display")
    except Exception as e:
        st.error(f"Error when creating best-selling products chart: {e}")

    # Order value distribution with Spark and Plotly
    st.subheader("💰 Order Value Distribution")
    
    try:
        # Calculate order values with Spark
        order_values_spark = filtered_spark_df.groupBy("InvoiceNo") \
            .agg(spark_sum("TotalValue").alias("TotalValue"))
        
        # Calculate 95th percentile to remove outliers
        percentile_df = order_values_spark.agg(
            percentile_approx("TotalValue", 0.95).alias("upper_limit")
        ).collect()[0]
        upper_limit = percentile_df.upper_limit
        
        # Filter orders below the upper limit
        filtered_order_values_spark = order_values_spark.filter(col("TotalValue") <= upper_limit)
        
        # Convert to Pandas for visualization
        filtered_order_values = filtered_order_values_spark.toPandas()
        
        # Create histogram with Plotly
        fig = px.histogram(
            filtered_order_values, 
            x='TotalValue',
            nbins=50,
            labels={'TotalValue': 'Order Value', 'count': 'Number of Orders'},
            title='Order Value Distribution (excluding top 5% outliers)',
            color_discrete_sequence=[main_color],
            template='plotly_white'
        )
        
        # Improve display
        fig.update_layout(
            xaxis_title='Order Value ($)',
            yaxis_title='Number of Orders',
            bargap=0.1,
            height=500
        )
        
        # Format currency on hover
        fig.update_traces(
            hovertemplate='Value: $%{x:.2f}<br>Count: %{y}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating order value distribution chart: {e}")

with tab2:
    st.markdown("""
        <h2 style='text-align: center; color: #2E86C1;'>🌍 Country Analysis</h2>
    """, unsafe_allow_html=True)
    
    # Revenue by country with Spark and Plotly
    try:
        # Calculate country metrics with Spark
        country_data_spark = filtered_spark_df.groupBy("Country").agg(
            spark_sum("TotalValue").alias("Revenue"),
            countDistinct("InvoiceNo").alias("Orders"),
            countDistinct("CustomerID").alias("Customers")
        )
        
        # Calculate average order value
        country_data_spark = country_data_spark.withColumn(
            "Avg_Order_Value", 
            col("Revenue") / col("Orders")
        )
        
        # Sort and limit
        top_countries_spark = country_data_spark.orderBy(desc("Revenue")).limit(10)
        
        # Convert to Pandas for visualization
        country_data = country_data_spark.toPandas()
        top_countries = top_countries_spark.toPandas()
        
        # Get number of countries to display
        display_countries = len(top_countries)
        
        if display_countries > 0:
            # Create layout with 2 columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Create revenue by country chart
                fig1 = px.bar(
                    top_countries, 
                    y='Country', 
                    x='Revenue',
                    orientation='h',
                    labels={'Country': 'Country', 'Revenue': 'Revenue'},
                    title=f'Top {display_countries} Countries by Revenue',
                    color='Revenue',
                    color_continuous_scale=px.colors.sequential.Blues,
                    template='plotly_white'
                )
                
                # Improve display
                fig1.update_layout(
                    xaxis_title='Revenue ($)',
                    yaxis_title='Country',
                    hovermode='y unified',
                    coloraxis_showscale=False,
                    height=500
                )
                
                # Add values to horizontal bars
                fig1.update_traces(
                    texttemplate='$%{x:.2f}',
                    textposition='outside',
                    hovertemplate='Country: %{y}<br>Revenue: $%{x:.2f}'
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Create orders by country chart
                fig2 = px.bar(
                    top_countries, 
                    y='Country', 
                    x='Orders',
                    orientation='h',
                    labels={'Country': 'Country', 'Orders': 'Number of Orders'},
                    title=f'Top {display_countries} Countries by Number of Orders',
                    color='Orders',
                    color_continuous_scale=px.colors.sequential.Greens,
                    template='plotly_white'
                )
                
                # Improve display
                fig2.update_layout(
                    xaxis_title='Number of Orders',
                    yaxis_title='Country',
                    hovermode='y unified',
                    coloraxis_showscale=False,
                    height=500
                )
                
                # Add values to horizontal bars
                fig2.update_traces(
                    texttemplate='%{x}',
                    textposition='outside',
                    hovertemplate='Country: %{y}<br>Orders: %{x}'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Create world map with revenue
            st.subheader("🗺️ Revenue on World Map")
            
            try:
                # Use the complete country data for the map
                fig_map = px.choropleth(
                    country_data,
                    locations='Country',
                    locationmode='country names',
                    color='Revenue',
                    hover_name='Country',
                    color_continuous_scale=px.colors.sequential.Blues,
                    title='Revenue by Country',
                    labels={'Revenue': 'Revenue ($)'}
                )
                
                # Improve display
                fig_map.update_layout(
                    geo=dict(
                        showcoastlines=True,
                        projection_type='natural earth',
                        showland=True,
                        landcolor='rgb(243, 243, 243)',
                        countrycolor='rgb(204, 204, 204)'
                    ),
                    height=600,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                # Format tooltip
                fig_map.update_traces(
                    hovertemplate='<b>%{hovertext}</b><br>Revenue: $%{z:.2f}'
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Error when creating world map: {e}")
            
            # Create average order value by country chart
            st.subheader("💵 Average Order Value by Country")
            
            # Sort by average order value
            top_avg_countries = country_data.sort_values('Avg_Order_Value', ascending=False).head(display_countries)
            
            fig3 = px.bar(
                top_avg_countries, 
                y='Country', 
                x='Avg_Order_Value',
                orientation='h',
                labels={'Country': 'Country', 'Avg_Order_Value': 'Average Order Value'},
                title=f'Top {display_countries} Countries by Average Order Value',
                color='Avg_Order_Value',
                color_continuous_scale=px.colors.sequential.Oranges,
                template='plotly_white'
            )
            
            # Improve display
            fig3.update_layout(
                xaxis_title='Average Order Value ($)',
                yaxis_title='Country',
                hovermode='y unified',
                coloraxis_showscale=False,
                height=500
            )
            
            # Add values to horizontal bars
            fig3.update_traces(
                texttemplate='$%{x:.2f}',
                textposition='outside',
                hovertemplate='Country: %{y}<br>Avg Order Value: $%{x:.2f}'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough country data to display")
    except Exception as e:
        st.error(f"Error when creating country analysis charts: {e}")
        
    # Add scatter plot to analyze relationship between orders and revenue by country
    st.subheader("📊 Relationship Between Orders and Revenue by Country")
    
    try:
        # Use data calculated above (country_data)
        if not country_data.empty:
            # Create scatter plot with Plotly
            fig_scatter = px.scatter(
                country_data,
                x='Orders', 
                y='Revenue',
                size='Customers',  # Point size by number of customers
                color='Avg_Order_Value',  # Color by average order value
                hover_name='Country',
                labels={
                    'Orders': 'Number of Orders', 
                    'Revenue': 'Revenue ($)',
                    'Customers': 'Number of Customers',
                    'Avg_Order_Value': 'Avg Order Value ($)'
                },
                title='Relationship Between Orders, Revenue and Customers by Country',
                color_continuous_scale=px.colors.sequential.Viridis,
                template='plotly_white'
            )
            
            # Improve display
            fig_scatter.update_layout(
                xaxis_title='Number of Orders',
                yaxis_title='Revenue ($)',
                coloraxis_colorbar=dict(title='Avg Order Value ($)'),
                height=600
            )
            
            # Customize tooltip
            fig_scatter.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>Orders: %{x}<br>Revenue: $%{y:.2f}<br>Customers: %{marker.size}<br>Avg Order Value: $%{marker.color:.2f}'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Not enough country data to display")
    except Exception as e:
        st.error(f"Error when creating scatter plot: {e}")

with tab3:
    st.markdown("""
        <h2 style='text-align: center; color: #2E86C1;'>⏱️ Time Analysis</h2>
    """, unsafe_allow_html=True)
    
    # Analysis by hour of day with Spark
    st.subheader("🕒 Orders by Hour of Day")
    
    try:
        # Calculate hourly metrics with Spark
        hourly_data_spark = filtered_spark_df.groupBy("Hour").agg(
            countDistinct("InvoiceNo").alias("Orders"),
            spark_sum("TotalValue").alias("TotalValue")
        ).orderBy("Hour")
        
        # Convert to Pandas for visualization
        hourly_data = hourly_data_spark.toPandas()
        
        # Create line chart with Plotly
        fig = go.Figure()
        
        # Add orders data
        fig.add_trace(go.Scatter(
            x=hourly_data['Hour'], 
            y=hourly_data['Orders'],
            mode='lines+markers',
            name='Orders',
            line=dict(color=main_color, width=3),
            marker=dict(size=8)
        ))
        
        # Add secondary y-axis for revenue
        fig.add_trace(go.Scatter(
            x=hourly_data['Hour'], 
            y=hourly_data['TotalValue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color=secondary_color, width=3, dash='dash'),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        # Update layout with two y-axes
        fig.update_layout(
            title='Orders and Revenue by Hour of Day',
            xaxis=dict(
                title='Hour of Day',
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                title='Number of Orders',
                titlefont=dict(color=main_color),
                tickfont=dict(color=main_color)
            ),
            yaxis2=dict(
                title='Revenue ($)',
                titlefont=dict(color=secondary_color),
                tickfont=dict(color=secondary_color),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating hourly chart: {e}")
    
    # Analysis by day of week with Spark
    st.subheader("📅 Revenue by Day of Week")
    
    try:
        # Create day names mapping
        day_names = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        
        # Calculate revenue by day of week with Spark
        weekday_data_spark = filtered_spark_df.groupBy("DayOfWeek") \
            .agg(spark_sum("TotalValue").alias("TotalValue")) \
            .orderBy("DayOfWeek")
        
        # Convert to Pandas for visualization
        weekday_data = weekday_data_spark.toPandas()
        
        # Add day names
        weekday_data['Weekday'] = weekday_data['DayOfWeek'].map(day_names)
        
        # Create bar chart with Plotly
        fig = px.bar(
            weekday_data, 
            x='Weekday', 
            y='TotalValue',
            labels={'Weekday': 'Day of Week', 'TotalValue': 'Revenue'},
            title='Revenue by Day of Week',
            color='TotalValue',
            color_continuous_scale=px.colors.sequential.Blues,
            template='plotly_white',
            category_orders={"Weekday": [day_names[i] for i in range(7)]}
        )
        
        # Improve display
        fig.update_layout(
            xaxis_title='Day of Week',
            yaxis_title='Revenue ($)',
            coloraxis_showscale=False,
            height=500
        )
        
        # Add values to bars
        fig.update_traces(
            texttemplate='$%{y:.2f}',
            textposition='outside',
            hovertemplate='Day: %{x}<br>Revenue: $%{y:.2f}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating day of week chart: {e}")
    
    # Time trend analysis with Spark
    st.subheader("📈 Revenue and Orders Trend Over Time")
    
    try:
        # Calculate daily metrics with Spark
        daily_data_spark = filtered_spark_df.withColumn(
            "Date", date_format(col("InvoiceDate"), "yyyy-MM-dd")
        ).groupBy("Date").agg(
            spark_sum("TotalValue").alias("Revenue"),
            countDistinct("InvoiceNo").alias("Orders")
        ).orderBy("Date")
        
        # Convert to Pandas for visualization
        daily_data = daily_data_spark.toPandas()
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        
        # Create time trend chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add revenue data
        fig.add_trace(
            go.Scatter(
                x=daily_data['Date'], 
                y=daily_data['Revenue'],
                mode='lines',
                name='Revenue',
                line=dict(color=main_color, width=3)
            ),
            secondary_y=False
        )
        
        # Add orders data
        fig.add_trace(
            go.Scatter(
                x=daily_data['Date'], 
                y=daily_data['Orders'],
                mode='lines',
                name='Orders',
                line=dict(color=secondary_color, width=3)
            ),
            secondary_y=True
        )
        
        # Update titles and labels
        fig.update_layout(
            title='Revenue and Orders Trend Over Time',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            template='plotly_white',
            height=600
        )
        
        # Update axis titles
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Revenue ($)', secondary_y=False)
        fig.update_yaxes(title_text='Number of Orders', secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating trend chart: {e}")
    
    # Heat map chart analyzing purchase patterns by hour and day of week with Spark
    st.subheader("🔥 Heat Map: Orders by Hour and Day of Week")
    
    try:
        # Calculate orders by hour and day of week with Spark
        heatmap_data_spark = filtered_spark_df.groupBy("DayOfWeek", "Hour") \
            .agg(countDistinct("InvoiceNo").alias("Orders"))
        
        # Convert to Pandas for visualization
        heatmap_data = heatmap_data_spark.toPandas()
        
        # Add day names
        heatmap_data['WeekdayName'] = heatmap_data['DayOfWeek'].map(day_names)
        
        # Create pivot table for heatmap
        pivot_data = heatmap_data.pivot(index='WeekdayName', columns='Hour', values='Orders')
        
        # Sort weekdays in correct order
        weekday_order = [day_names[i] for i in range(7)]
        pivot_data = pivot_data.reindex(weekday_order)
        
        # Create heatmap with Plotly
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Number of Orders"),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='Blues',
            aspect='auto',
            title='Heat Map: Orders by Hour and Day of Week'
        )
        
        # Improve display
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            height=500
        )
        
        # Show values in cells
        fig.update_traces(
            text=pivot_data.values,
            texttemplate="%{text}",
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Orders: %{z}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating heat map: {e}")
    
    # Analysis by month of year with Spark
    st.subheader("🗓️ Revenue by Month of Year")
    
    try:
        # Calculate revenue by month and year with Spark
        monthly_data_spark = filtered_spark_df.groupBy("Year", "Month") \
            .agg(spark_sum("TotalValue").alias("TotalValue")) \
            .orderBy("Year", "Month")
        
        # Convert to Pandas for visualization
        monthly_data = monthly_data_spark.toPandas()
        
        # Month names
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        
        # Add month names
        monthly_data['MonthName'] = monthly_data['Month'].map(month_names)
        
        # Create new column for x-axis
        monthly_data['MonthYear'] = monthly_data.apply(lambda x: f"{x['MonthName']} {int(x['Year'])}", axis=1)
        
        # Create bar chart with Plotly
        fig = px.bar(
            monthly_data, 
            x='MonthYear', 
            y='TotalValue',
            labels={'MonthYear': 'Month - Year', 'TotalValue': 'Revenue'},
            title='Revenue by Month of Year',
            color='Year',
            color_discrete_sequence=px.colors.qualitative.Set1,
            template='plotly_white'
        )
        
        # Improve display
        fig.update_layout(
            xaxis_title='Month - Year',
            yaxis_title='Revenue ($)',
            xaxis=dict(tickangle=45),
            legend_title='Year',
            height=500
        )
        
        # Format tooltip
        fig.update_traces(
            hovertemplate='%{x}<br>Revenue: $%{y:.2f}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating monthly chart: {e}")
    
    # Add growth rate chart using Spark Window functions
    st.subheader("📊 Monthly Revenue Growth Rate")
    
    try:
        # Phân tích theo tháng và quốc gia
        monthly_country_data_spark = filtered_spark_df.groupBy("Year", "Month", "Country") \
            .agg(spark_sum("TotalValue").alias("TotalValue")) \
            .orderBy("Country", "Year", "Month")

        # Tỷ lệ tăng trưởng sử dụng hàm Window theo quốc gia
        window_spec = Window.partitionBy("Country").orderBy("Year", "Month")
        growth_data_spark = monthly_country_data_spark.withColumn(
            "PrevRevenue", lag("TotalValue").over(window_spec)
        ).withColumn(
            "GrowthRate",
            when(col("PrevRevenue").isNotNull(), 
                (col("TotalValue") - col("PrevRevenue")) / col("PrevRevenue") * 100
            ).otherwise(None)
        )
        
        # Convert to Pandas for visualization
        growth_data = growth_data_spark.toPandas()
        
        # Remove first row as it has no previous month data
        growth_data = growth_data.dropna()
        
        # Add month-year information
        growth_data['MonthName'] = growth_data['Month'].map(month_names)
        growth_data['MonthYear'] = growth_data.apply(lambda x: f"{x['MonthName']} {int(x['Year'])}", axis=1)
        
        # Create chart with Plotly
        fig = go.Figure()
        
        # Add positive growth data
        positive_growth = growth_data[growth_data['GrowthRate'] >= 0]
        negative_growth = growth_data[growth_data['GrowthRate'] < 0]
        
        if not positive_growth.empty:
            fig.add_trace(go.Bar(
                x=positive_growth['MonthYear'],
                y=positive_growth['GrowthRate'],
                name='Positive Growth',
                marker_color='green'
            ))
        
        if not negative_growth.empty:
            fig.add_trace(go.Bar(
                x=negative_growth['MonthYear'],
                y=negative_growth['GrowthRate'],
                name='Negative Growth',
                marker_color='red'
            ))
        
        # Add 0% reference line
        fig.add_shape(
            type='line',
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            xref='paper',
            yref='y',
            line=dict(color='black', width=1.5, dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title='Monthly Revenue Growth Rate (%)',
            xaxis_title='Month - Year',
            yaxis_title='Growth Rate (%)',
            xaxis=dict(tickangle=45),
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        # Format tooltip
        fig.update_traces(
            hovertemplate='%{x}<br>Growth Rate: %{y:.2f}%'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error when creating growth rate chart: {e}")

   
    # RFM Analysis with Spark
    st.subheader("👥 Customer RFM Analysis")
    
    try:
        # Filter data with valid CustomerID
        customer_df_spark = filtered_spark_df.filter(col("CustomerID").isNotNull())
        
        if customer_df_spark.count() > 0:
            # Get the maximum date for Recency calculation
            max_date_row = customer_df_spark.agg(functions.max("InvoiceDate").alias("max_date")).collect()[0]
            max_date = max_date_row["max_date"]
            
            # Calculate RFM metrics with Spark
            rfm_spark = customer_df_spark.groupBy("CustomerID").agg(
                datediff(lit(max_date), functions.max("InvoiceDate")).alias("Recency"),
                countDistinct("InvoiceNo").alias("Frequency"),
                spark_sum("TotalValue").alias("Monetary")
            )
            
            # Convert to Pandas for scoring (easier in Pandas)
            rfm = rfm_spark.toPandas()
            
            # Create RFM segments
            # Simplified approach: divide each metric into 3 segments
            rfm['R_Score'] = pd.qcut(rfm['Recency'], 3, labels=[3, 2, 1])  # Lower is better for recency
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 3, labels=[1, 2, 3])  # Higher is better
            rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 3, labels=[1, 2, 3])  # Higher is better
            
            # Convert to numeric
            rfm['R_Score'] = rfm['R_Score'].astype(int)
            rfm['F_Score'] = rfm['F_Score'].astype(int)
            rfm['M_Score'] = rfm['M_Score'].astype(int)
            
            # Calculate RFM Score
            rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
            
            # Create segments
            rfm['Segment'] = pd.cut(
                rfm['RFM_Score'],
                bins=[0, 3, 6, 9],
                labels=['Low Value', 'Mid Value', 'High Value']
            )
            
            # Display segment distribution
            segment_counts = rfm['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            # Create chart
            fig = px.bar(
                segment_counts, 
                x='Segment', 
                y='Count',
                color='Segment',
                title='Customer Segment Distribution',
                color_discrete_sequence=px.colors.qualitative.Safe,
                template='plotly_white'
            )
            
            # Improve display
            fig.update_layout(
                xaxis_title='Customer Segment',
                yaxis_title='Number of Customers',
                showlegend=False,
                height=500
            )
            
            # Format tooltip
            fig.update_traces(
                texttemplate='%{y}',
                textposition='outside',
                hovertemplate='Segment: %{x}<br>Customers: %{y}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 3D scatter plot for RFM
            st.subheader("3D RFM Customer Segmentation")
            
            # Sample data to avoid performance issues with large datasets
            if len(rfm) > 1000:
                sample_rfm = rfm.sample(1000, random_state=42)
            else:
                sample_rfm = rfm
                
            # Create 3D scatter plot
            fig = px.scatter_3d(
                sample_rfm, 
                x='Recency', 
                y='Frequency', 
                z='Monetary',
                color='Segment',
                size='RFM_Score',
                opacity=0.7,
                title='3D RFM Customer Segmentation',
                color_discrete_sequence=px.colors.qualitative.Safe,
                template='plotly_white'
            )
            
            # Improve display
            fig.update_layout(
                scene=dict(
                    xaxis_title='Recency (days)',
                    yaxis_title='Frequency (orders)',
                    zaxis_title='Monetary (value $)'
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough customer data to perform RFM analysis.")
    except Exception as e:
        st.error(f"Error when creating RFM analysis: {e}")


with tab4:
    st.markdown("""
        <h2 style='text-align: center; color: #2E86C1;'>🔗 Associate Analysis</h2>
    """, unsafe_allow_html=True)
    
    # Load Association Rules Data
    try:
        st.subheader("📊 Association Rules Analysis")
        
        # Read Association Rules data
        @st.cache_data(show_spinner=False)
        def load_association_rules():
            try:
                # Read the file using pandas
                rules_df = pd.read_csv(r'C:\Users\Admin\Desktop\big_data_project\associate_retail_data\fpgrowth_results_max30items\association_rules_with_desc.csv', encoding='utf-8')
                return rules_df
            except Exception as e:
                st.error(f"Error loading association rules data: {e}")
                return None
        
        rules_df = load_association_rules()
        
        if rules_df is not None:
            st.info(f"Loaded {len(rules_df)} association rules")
            
            # 1. Top 20 rules by lift
            st.subheader("🔝 Top 20 Rules by Lift")
            
            # Sort by lift and take top 20
            top_lift_rules = rules_df.sort_values('lift', ascending=False).head(20)
            
            # Create a better visual representation of rules
            top_lift_rules['rule'] = top_lift_rules.apply(
                lambda x: f"{x['antecedent_with_desc']} → {x['consequent_with_desc']}", 
                axis=1
            )
            
            # Create horizontal bar chart
            fig_lift = px.bar(
                top_lift_rules, 
                y='rule', 
                x='lift',
                orientation='h',
                labels={'rule': 'Association Rule', 'lift': 'Lift'},
                title='Top 20 Association Rules by Lift',
                color='lift',
                color_continuous_scale=px.colors.sequential.Blues,
                template='plotly_white'
            )
            
            # Improve display
            fig_lift.update_layout(
                xaxis_title='Lift',
                yaxis_title='Association Rule',
                yaxis={'categoryorder':'total ascending'},  # Sort y-axis by x values
                hovermode='y unified',
                height=600
            )
            
            # Add extra information on hover
            fig_lift.update_traces(
                hovertemplate='<b>%{y}</b><br>Lift: %{x:.2f}<br>Confidence: %{customdata[0]:.2f}<br>Support: %{customdata[1]:.4f}',
                customdata=top_lift_rules[['confidence', 'support']].values
            )
            
            st.plotly_chart(fig_lift, use_container_width=True)
            
            # 2. Top 20 rules by confidence
            st.subheader("🔝 Top 20 Rules by Confidence")
            
            # Sort by confidence and take top 20
            top_conf_rules = rules_df.sort_values('confidence', ascending=False).head(20)
            
            # Create a better visual representation of rules
            top_conf_rules['rule'] = top_conf_rules.apply(
                lambda x: f"{x['antecedent_with_desc']} → {x['consequent_with_desc']}", 
                axis=1
            )
            
            # Create horizontal bar chart
            fig_conf = px.bar(
                top_conf_rules, 
                y='rule', 
                x='confidence',
                orientation='h',
                labels={'rule': 'Association Rule', 'confidence': 'Confidence'},
                title='Top 20 Association Rules by Confidence',
                color='confidence',
                color_continuous_scale=px.colors.sequential.Greens,
                template='plotly_white'
            )
            
            # Improve display
            fig_conf.update_layout(
                xaxis_title='Confidence',
                yaxis_title='Association Rule',
                yaxis={'categoryorder':'total ascending'},  # Sort y-axis by x values
                hovermode='y unified',
                height=600
            )
            
            # Add extra information on hover
            fig_conf.update_traces(
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}<br>Lift: %{customdata[0]:.2f}<br>Support: %{customdata[1]:.4f}',
                customdata=top_conf_rules[['lift', 'support']].values
            )
            
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Load Frequent Itemsets Data
            @st.cache_data(show_spinner=False)
            def load_frequent_itemsets():
                try:
                    # Read the file using pandas 
                    itemsets_df = pd.read_csv(r'C:\Users\Admin\Desktop\big_data_project\associate_retail_data\fpgrowth_results_max30items\frequent_itemsets_with_desc.csv', encoding='utf-8')
                    return itemsets_df
                except Exception as e:
                    st.error(f"Error loading frequent itemsets data: {e}")
                    return None
            
            itemsets_df = load_frequent_itemsets()
            
            if itemsets_df is not None:
                st.info(f"Loaded {len(itemsets_df)} frequent itemsets")
                
                # 3. Top 20 frequent itemsets
                st.subheader("🔝 Top 20 Frequent Itemsets")
                
                # Sort by frequency and take top 20
                top_frequent_items = itemsets_df.sort_values('freq', ascending=False).head(20)
                
                # Some items_with_desc might be very long, let's truncate them for display
                top_frequent_items['items_display'] = top_frequent_items['items_with_desc'].apply(
                    lambda x: (x[:75] + '...') if len(x) > 75 else x
                )
                
                # Create horizontal bar chart
                fig_freq = px.bar(
                    top_frequent_items, 
                    y='items_display', 
                    x='freq',
                    orientation='h',
                    labels={'items_display': 'Itemset', 'freq': 'Frequency'},
                    title='Top 20 Frequent Itemsets',
                    color='freq',
                    color_continuous_scale=px.colors.sequential.Oranges,
                    template='plotly_white'
                )
                
                # Improve display
                fig_freq.update_layout(
                    xaxis_title='Frequency',
                    yaxis_title='Itemset',
                    yaxis={'categoryorder':'total ascending'},  # Sort y-axis by x values
                    hovermode='y unified',
                    height=600
                )
                
                # Add item codes on hover
                fig_freq.update_traces(
                    hovertemplate='<b>Items:</b> %{customdata[0]}<br><b>Frequency:</b> %{x}',
                    customdata=top_frequent_items[['items']].values
                )
                
                st.plotly_chart(fig_freq, use_container_width=True)
                
                # Display raw data in tabs
                st.subheader("📋 Raw Data")
                
                raw_tab1, raw_tab2 = st.tabs(["Association Rules", "Frequent Itemsets"])
                
                with raw_tab1:
                    st.dataframe(rules_df, use_container_width=True)
                    
                    # Add download button
                    csv_rules = rules_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Association Rules Data",
                        data=csv_rules,
                        file_name="association_rules.csv",
                        mime="text/csv",
                    )
                
                with raw_tab2:
                    st.dataframe(itemsets_df, use_container_width=True)
                    
                    # Add download button
                    csv_itemsets = itemsets_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Frequent Itemsets Data",
                        data=csv_itemsets,
                        file_name="frequent_itemsets.csv",
                        mime="text/csv",
                    )
            else:
                st.error("Could not load frequent itemsets data.")
        else:
            st.error("Could not load association rules data.")
    except Exception as e:
        st.error(f"Error in association analysis tab: {e}")
# Add footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: gray;'>Online Retail Dashboard • PySpark Processing • Created with Streamlit and Plotly</p>
""", unsafe_allow_html=True)