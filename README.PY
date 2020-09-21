locations = ["HEAT - A_final.xls","HEAT - B_final.xls","HEAT - C_final.xls","HEAT - D_final.xls", "HEAT - E_final.xls"]
df_A = pd.read_excel("HEAT - A_final.xls", header = 3, skiprows = range(4,5))
df_B = pd.read_excel("HEAT - B_final.xls", header = 3, skiprows = range(4,5))
df_C = pd.read_excel("HEAT - C_final.xls", header = 3, skiprows = range(4,5))
df_D = pd.read_excel("HEAT - D_final.xls", header = 3, skiprows = range(4,5))
df_E = pd.read_excel("HEAT - E_final.xls", header = 3, skiprows = range(4,5))

#### FOR THIS FIRST PART I USED THE PANDAS TO READ THE EXCEL FILE
#### AND PUT EACH ONE TO THEIR SPECIFIC DATAFRAME
#### THEN SEPARATED SOME IMPORTANT VARIABLES FROM ALL OF THE DATAFRAMES
#### ALSO SEPARATED THE SPECIFIC VARIABLES TO CALL LATER ON

Temperature = (df_A["Temperature"], df_B["Temperature"], df_C["Temperature"], df_D["Temperature"], df_E["Temperature"])
TrueDirection = (df_A["Direction ‚ True"], df_B["Direction ‚ True"], df_C["Direction ‚ True"], df_D["Direction ‚ True"], df_E["Direction ‚ True"])
WindSpeed = (df_A["Wind Speed"], df_B["Wind Speed"], df_C["Wind Speed"], df_D["Wind Speed"], df_E["Wind Speed"])
Wetbulb = (df_A["NA Wet Bulb Temperature"], df_B["NA Wet Bulb Temperature"], df_C["NA Wet Bulb Temperature"], df_D["NA Wet Bulb Temperature"], df_E["NA Wet Bulb Temperature"])
CrossWind = (df_A["Crosswind Speed"], df_B["Crosswind Speed"], df_C["Crosswind Speed"], df_D["Crosswind Speed"], df_E["Crosswind Speed"])
WBGT = (df_A["WBGT"], df_B["WBGT"], df_C["WBGT"], df_D["WBGT"], df_E["WBGT"])

EMP_A=df_A["Temperature"]
TEMP_B=df_B["Temperature"]
TEMP_C=df_C["Temperature"]
TEMP_D=df_D["Temperature"]
TEMP_E=df_E["Temperature"]

DIR_A=df_A["Direction ‚ True"]
DIR_B=df_B["Direction ‚ True"]
DIR_C=df_C["Direction ‚ True"]
DIR_D=df_D["Direction ‚ True"]
DIR_E=df_E["Direction ‚ True"]

WINDSP_A=df_A["Wind Speed"]
WINDSP_B=df_B["Wind Speed"]
WINDSP_C=df_C["Wind Speed"]
WINDSP_D=df_D["Wind Speed"]
WINDSP_E=df_E["Wind Speed"]

WBGT_A=df_A["WBGT"] 
WBGT_B=df_B["WBGT"] 
WBGT_C=df_C["WBGT"] 
WBGT_D=df_D["WBGT"] 
WBGT_E=df_E["WBGT"] 

CROSS_A = df_A["Crosswind Speed"]   
CROSS_B = df_B["Crosswind Speed"]  
CROSS_C = df_C["Crosswind Speed"]  
CROSS_D = df_D["Crosswind Speed"]  
CROSS_E = df_E["Crosswind Speed"]
################ A1 ########################
### 1- A SIMPLE MEAN, VARIANCE AND STANDARD DEVIATION PRINT FOR THE DATAFRAMES

Mean = (df_A.mean(), df_B.mean(), df_C.mean(),df_D.mean(), df_E.mean())
Variance = (df_A.var(), df_B.var(), df_C.var(),df_D.var(), df_E.var())
Standard_Deviation = (df_A.std(), df_B.std(), df_C.std(),df_D.std(), df_E.std())
print(Mean, Variance, Standard_Deviation)

### 2- FOR THIS I CREATED A FUNCTION THAT USES EACH OF THE FORMENTIONED SEPARATED VARIABLES, 
### IN THIS CASE TEMPERATURE AND DEFINED THE NUMBER OF BINS IN B AND C, WITH D INDICATING THE SENSOR.


### 3 - BY MODIFYING THE FREQUENCY CODE FROM THE LECTURE I MADE A FUNCTION THAT
### WILL PLOT THE FREQUENCIES OF THE TEMPERATURES ALL TOGETHER IN THE SAME PLOT.

### 4 - SIMPLY MADE USE OF THE BOXPLOT FUNCTION WITH THE SEPARATED SPECIFIC VARIABLES

######################## A2 ############################
### 1 - Plot PMF, PDF and CDF for the 5 sensors Temperature values in independent plots (or subplots).
### HERE I CREATED A FUNCTION BY MODYFING THE CODE FROM THE LECTURE TO CREATE THE PMFS.
### USING THE SUBPLOTS TO PRODUCE ONE IMAGE FOR ALL OF THE TEMPERATURES,

### FOR THE PDFS I SET THE DETERMINED NUMBER OF BINS TO 27 AND DIDNT USE THE VALUE COUNTS.
### I USED THE SNS.DISTPLOT TO SHOW THE CONTINUOUS DATA WITH THE .HIST AND SETTING THE DENSITY FOR BETTER VISUALIZATION

### FOR THE CDFS, I CREATED A FUNCTION BY ALSO MODIFIED THE CODE FROM THE LECTURES
### WHERE THE INPUT a,b,c,d,e IS THE SAME AS THE OTHER GRAPHS IN THIS QUESTION.

######################## A3 ############################
### 1 - HERE I USED THE SAME FUNCTION AS BEFORE, ONLY REPLACING THE DESCRIPTIVE LABELS WITH WINDSPEED
### ALSO CHANGING THE INPUT DATAFRAME AT THE END

### FOR THE KDE I UED ALMOT THE SAME FUNCTION AS A PDF, ONLY CHANGING THE COLOUR AND
### kde = true, TO SHOW THE KDE VALUES

### 2 - TO COMPUTE THE CORRELATIONS I HAD TO MAKE A FUNCTION THAT WOULD
### INTERPOLATE THE DATA BETWEEN ALL OF THE SENSORS
### THEN I NORMALIZED THEM BY USING THE FORMULA PRESENTED IN THE CODE
### AFTERWARDS IT WAS JUST A MATTER OF WRITING ALL OF THE PAIRINGS WITH THE STATS.PEARSONR,
### STATS.SPEARMANR AND USING THE SCATTERPLOTS FROM .PLT

####################### A4 #############################
### 1 - REPEATED THE CDF FUNCTION FOR TEMPERATURE AND WINDSPEED


### 2 - HERE I CREATED A FUNCTION TO DETERMINE A CONFIDENCE INTERVAL
### USING THE TEMPERATURE AND WINDSPEED DATAFRAMES FROM EACH TABLE.
### AFTERWARDS MADE IT SO IT WILL CREATE AND SAVE THE CONFIDENCE INTERVALS
### TO A CSV. FILE.

### 3 - CREATED A FUNCTION THAT USES THE DATA FROM THE SPECIFIC TEMPERATURE AND WINDSPEED VALUES OF EACH DATAFRAME.
### USING THE STATS.TTEST_IND TO CALCULATE THE P-VALUES AND USING THE PRINT TO SEE THEM.

### THAT'S ALL FOLKS

