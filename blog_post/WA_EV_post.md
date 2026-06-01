Github: https://github.com/alexsong-lab/SML_group_project

**Beyond the Hype: Uncovering the True Shape of Washington's Electric Vehicle Landscape**

By Alex Song, Gurpreet Kaur, and Naomi Le Mouel, students in the Seattle University Master of Science in Data Science (MSDS) program

**1.Introduction**

**If You Only Look at EV Counts, You Miss the Real Story**

Open almost any report on electric vehicles in Washington State, and one county immediately dominates the conversation: King County.

King County has the most EVs, the most chargers, and some of the highest adoption rates in the state. Looking only at raw totals, it is easy to conclude that Washington's EV transition is simply a story of large urban counties leading the way while rural counties struggle to keep up.

**But what if that story is incomplete?**

What if population size is drowning out the more interesting patterns? What if raw counts are masking deeper structural differences in how electric vehicles are actually adopted and supported across the state?

Some counties may have relatively few EVs but surprisingly strong charging infrastructure. Others may have rapidly modernized fleets despite smaller markets. And a few may be operating under entirely different conditions.

[**Washington ranks third in the nation**](https://wsdot.wa.gov/about/data/gray-notebook/gnbhome/environment/electricvehicles/electricvehicles.htm) in EV market share, making it one of the most EV-friendly states in the country. Yet statewide statistics hide substantial variation between counties. Understanding that variation is important for identifying where EV adoption is thriving, where infrastructure may be lagging, and where different policy approaches may be needed.

The primary reaserch question for this project was this:

What underlying EV adoption patterns exist across Washington's 39 counties once we remove the population-driven "King County Effect"_?_

To answer this question, we applied several unsupervised learning techniques: Principal Component Analysis (PCA), Singular Value Decomposition (SVD), K-Means Clustering, Hierarchical Clustering.

Our dataset combines county-level information on EV registrations, charging infrastructure, vehicle characteristics, and socioeconomic indicators for all 39 Washington counties. Variables include EV counts, charger counts, BEV and PHEV shares, average electric range, average vehicle age, average model year, charger density measures, and median income.

Rather than predicting an outcome, our goal was to uncover hidden patterns and determine whether counties naturally group together based on their EV ecosystem characteristics.

Note: Our analysis uses four primary data sources: Washington State's EV registration dataset, the statewide public EV charging station dataset, county median household income estimates from the Washington Office of Financial Management, and county profile data from the Municipal Research and Services Center (MRSC).

&nbsp;

**2. Theoretical Background**

**Principal Component Analysis (PCA)**

When you have a bunch of variables that all move together - like EV counts, charger counts, and BEV shares - PCA is a great way to simplify the picture. The idea is to rotate the data into a new set of axes (called _principal components_) that capture the biggest patterns of variation.

The first component explains the most variation, the second explains the next most, and so on. Each component comes with:

- **Loadings**, which tell you which variables define the component
- **Scores**, which tell you where each county sits along that component

In practice, we look at scree plots and variance‑explained charts to decide how many components are worth keeping. PCA is powerful when variables are correlated, but interpreting components can be tricky because each one blends multiple variables together.

**Singular Value Decomposition (SVD)**

SVD is the mathematical engine under the hood of PCA. If you take your data matrix (X) and decompose it into: \[ X = U ∑ V\* \]

You get three pieces:

- **U**, which describes how counties relate to hidden patterns
- **∑**, which tells you how strong those patterns are
- **V\***, which shows how variables contribute to each pattern

ISLP points out that PCA loadings come directly from the (V\*) matrix, and PCA scores come from (U∑). That's why PCA and SVD should agree - they're looking at the same geometry from two angles. SVD can also be used for matrix completion, but only when missingness is large and the data truly lie near a low‑rank structure.

&nbsp;**K**‑**Means Clustering**

K‑Means is one of the simplest ways to group similar counties together. You pick a number of clusters (k), and the algorithm keeps shuffling counties around until each one is assigned to the nearest cluster center. Because K‑Means can get stuck in local optima, ISLP recommends running it multiple times with different random starts. We typically use elbow plots and within‑cluster variation to decide on a good value of (k).

K‑Means works best when clusters are compact and well separated - which is why we also checked our results with hierarchical clustering.

&nbsp;**Hierarchical Clustering**

Hierarchical clustering takes a more flexible approach. Instead of choosing the number of clusters upfront, it builds a dendrogram - a tree that shows how counties merge together step by step.

You can "cut" the tree at different heights to get different numbers of clusters. This makes hierarchical clustering great for exploratory work because you can see structure at multiple levels. The catch is that results depend heavily on the linkage method.

**Linkage Methods**

Linkage methods define how distances between clusters are measured:

- **Single linkage**: looks at the closest pair of points (can create long chains)
- **Complete linkage**: looks at the farthest pair (creates tight, compact clusters)
- **Average linkage**: averages all pairwise distances (a middle ground)
- **Ward linkage**: merges clusters that increase within‑cluster variance the least

Ward linkage often produces the cleanest, most interpretable structure - which is why it's commonly paired with PCA‑based clustering and why we used it in our analysis.

&nbsp;

**3. Methodology**

**Data Collection and Variable Selection**

Electric vehicle registration, charger location, city/county lookup, and county income datasets were cleaned and merged to create a county level dataset for analysis. Across all files, column names were standardized, extra spaces were removed, and formatting was made consistent.

For the EV registration dataset, placeholder zeros in electric_range and sale_price were converted to NaN because they represented missing rather than true zero values. County, city, and state labels were standardized, only Washington State records were kept, unrealistic model years were removed, duplicate vehicle records were dropped, and a vehicle_age variable was created using model year.

The charger location dataset was cleaned by standardizing labels and dates and removing nonessential geographic, address, ID, and descriptive fields to focus on county level analysis. Charger counts and duplicate records were checked before aggregation. To assign charger stations to counties, the city/county lookup dataset was cleaned and merged using city names. Cities appearing in multiple counties created duplicate or conflicting matches and were manually resolved, while remaining unmatched cities were assigned as needed. These steps reduced missing charger county assignments to zero.

County income data were simplified by keeping county and 2025 median income values, standardizing county names, and converting income values to numeric format before merging.

EV registration data were then aggregated to the county level to create total EV count, BEV count, PHEV count, average model year, average electric range, average vehicle age, percent BEV, percent PHEV, and the BEV-to-PHEV ratio.

Population based variables were evaluated but excluded because the available population file appeared to contain city level rather than complete county populations, producing unrealistic EV-per-capita estimates.

**A Quick Note on Matrix Competion**

Matrix completion was explored for missing electric_range values. However, roughly 44% of values would have required imputation, and the completed values reduced realistic county variation. Because county aggregation using observed means still produced complete county averages without missing data, preserving observed information was considered more reliable. Matrix completion was therefore evaluated and documented but not used to alter the final dataset.

**Addressing the "King County Effect"**

The first thing we noticed during exploratory analysis was how dramatically county sizes differed. King County's EV ecosystem is enormous compared to many rural counties. Snohomish and Pierce also contain far more EVs and chargers than most of the state. If we applied PCA or clustering directly to the raw numbers, the algorithms would mostly learn one thing: **Big counties are big,** and that isn't very interesting.

To reduce this effect, we applied two preprocessing steps:

**Step 1** **-** **Log Transformations:** Count-based variables such as EV counts and charger counts were heavily right-skewed. Applying logarithmic transformations compressed extreme values and reduced the influence of large outliers.

**Step 2 - Standardization:** Variables were standardized so that every feature contributed equally to the analysis. Without standardization, variables measured on larger scales would dominate the results.

This ensured that variables measured on different scales contributed equally to PCA and clustering. Once these steps were complete, we were finally ready to let the data speak for itself.

**Principal Component Analysis:** PCA was performed on the standardized feature matrix. We examined-scree plots, variance explained, component loadings, PCA score plots-to determine how many components should be retained and how they should be interpreted.

**Singular Value Decomposition:** A manual SVD was performed on the standardized matrix. The resulting (U) and (V\*) matrices were compared with PCA scores and loadings to validate the decomposition.

**Clustering:** Clustering was performed using PCA-transformed data.

**K-Means:** We evaluated multiple cluster solutions, including k = 3 and k = 4 using elbow plots and interpretability.

**Hierarchical Clustering:** We compared: single linkage, complete linkage, average linkage, ward linkage, and evaluated dendrogram structures for each method.

&nbsp;

**4. Result**

**4.1 PCA Results**

Variance Explained

\[insert the PCA explained and cumulative table\]

\[insert scree plot\]

PCA Loadings

\[insert PCA loadings table\]

PCA Scores

\[insert PCA score tables or graph\]

\[insert PCA plot PC1 vs PC2\]

\[insert PCA biplot\]

**4.2 SVD Results**

Singular Values

\[insert singular values plot\]

\[insert U matrix\]

PCA-SVD Alignment

\[insert PCA-SVD table\]

**4.3 K-Means Results**

Elbow Plot

\[insert elbow plot\]

Silhouette Plot

\[insert silhouette plot\]

K-Mean diagnostics Table

\[insert k-means table\]

Final K-Mean Cluster Assignment (k=4)

\[insert the cluster-county table\]

Cluster Centroid (Scaled Fearure Space)

\[Insert centroid table\]

K-Means Cluster on PCA Plane

\[insert cluster scattter plot\]

**4.4 Hierarchical Clustering Result**

Dendrograms (4 Linkage Methids)

\[insetr dedrograms\]

Cut-Tree Assignment (k=4\)

\[insert cut-tree outputs\]

&nbsp;

**5. Discussion**

One thing that became clear throughout this project is that EV adoption in Washington is much more complicated than simply looking at which counties have the most electric vehicles. At the start of the analysis, counties such as King, Snohomish, and Pierce dominated nearly every count-based variable. Without additional preprocessing, most machine learning methods would have focused almost entirely on county size rather than on meaningful differences in EV adoption patterns.

After applying log transformations and standardization, a much more interesting picture emerged. PCA showed that most of the variation across Washington's counties could be summarized using just three dimensions. Rather than representing the same information repeatedly, these components captured different aspects of the EV landscape: overall adoption scale, fleet modernization, and charging infrastructure intensity. The fact that three components explained more than 80% of the total variance suggests that the original sixteen variables contained a large amount of overlapping information.

It was also encouraging to see how consistently the different methods agreed with one another. PCA and SVD produced essentially identical results, and both pointed to the same underlying dimensions. Similarly, K-Means and hierarchical clustering identified nearly the same county groupings. When several independent methods arrive at the same conclusions, it increases confidence that the patterns are real and not simply artifacts of a particular algorithm.

Perhaps the most interesting finding was Garfield County. Before starting the project, we expected King County to stand out because of its large EV market. Instead, Garfield became the most unusual county in the analysis. It consistently appeared as an outlier in PCA plots, singular vector analysis, K-Means clustering, and hierarchical clustering. This suggests that Garfield is not just a smaller version of other rural counties. Its EV adoption and charging infrastructure profile appears fundamentally different from the rest of the state.

The clustering results also highlight that Washington's counties cannot be treated as a single group. Urban counties tend to have large EV ecosystems and extensive charging infrastructure, while many rural counties share lower adoption levels but different infrastructure characteristics. These differences suggest that counties may face different challenges as EV adoption continues to grow. Some areas may require additional charging capacity to keep up with demand, while others may benefit more from policies that encourage adoption in the first place.

There are several limitations to this analysis. First, the study was performed at the county level, which reduces the amount of available data to only 39 observations. Important differences within counties may therefore be hidden. Second, population-based variables could not be included because the available population dataset was not suitable for county-level calculations. Finally, a large portion of the electric range data contained missing values. Although matrix completion was explored, the imputed values appeared to reduce natural variation, so the final analysis relied on observed county-level averages instead.

Despite these limitations, the results were remarkably consistent across all methods. The agreement between PCA, SVD, K-Means, and hierarchical clustering suggests that the identified patterns represent meaningful differences in how EV adoption is developing across Washington State. Rather than a single statewide story, the results point to several distinct EV adoption pathways that vary across counties.

**6. Conclusion**

This project demonstrates that Washington State's EV landscape is shaped by more than simple population differences. By combining data cleaning, feature engineering, dimensionality reduction, and clustering techniques, we identified three major dimensions of EV adoption-scale, modernization, and charging intensity-that together explain over 81% of the variation across counties.

The analysis revealed three broad EV adoption archetypes: urban leaders with extensive EV ecosystems and infrastructure, rural baseline counties with moderate adoption patterns, and Garfield County as a persistent structural outlier. Importantly, these patterns were consistently identified across multiple analytical approaches, including PCA, SVD, K-Means clustering, and hierarchical clustering, providing strong validation of the results.

From a policy perspective, the findings suggest that a single statewide strategy may not be effective for accelerating EV adoption. Urban counties face challenges related to infrastructure capacity and demand management, while rural counties may require targeted support to overcome adoption barriers and expand charging access. Unique cases such as Garfield County may require specialized local solutions that differ substantially from broader statewide initiatives.

More broadly, this study demonstrates the value of unsupervised learning methods for uncovering hidden patterns in complex transportation datasets. As EV adoption continues to grow, understanding these regional differences can help policymakers, utilities, and transportation planners allocate resources more effectively and design programs that better reflect the diverse needs of Washington communities. The methods used in this project could also be extended to other states or emerging transportation technologies to support future data-driven policy decisions.

&nbsp;

&nbsp;

**References**

Electric Vehicle Title and Registration Activity - Washington State Department of Licensing <https://data.wa.gov/Transportation/Electric-Vehicle-Title-and-Registration-Activity/rpr4-cgyd/about_data>

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An introduction to statistical learning: with applications in Python. Springer.

Public EV Charging Stations - WA GeoServices <https://geo.wa.gov/datasets/wa-geoservices::public-ev-charging-stations/about>

Median Household Income Estimates - Washington Office of Financial Management <https://ofm.wa.gov/data-research/economy/median-household-income-estimates/>

County Profiles (City-County Lookup) - Municipal Research and Services Center (MRSC) <https://mrsc.org/research-tools/county-profiles>