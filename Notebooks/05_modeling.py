#!/usr/bin/env python
# coding: utf-8

# # 5 Modeling<a id='5_Modeling'></a>

# ## 5.1 Contents<a id='5.1_Contents'></a>
# * [5 Modeling](#5_Modeling)
#   * [5.1 Contents](#5.1_Contents)
#   * [5.2 Introduction](#5.2_Introduction)
#   * [5.3 Imports](#5.3_Imports)
#   * [5.4 Load Model](#5.4_Load_Model)
#   * [5.5 Load Data](#5.5_Load_Data)
#   * [5.6 Refit Model On All Available Data (excluding Big Mountain)](#5.6_Refit_Model_On_All_Available_Data_(excluding_Big_Mountain))
#   * [5.7 Calculate Expected Big Mountain Ticket Price From The Model](#5.7_Calculate_Expected_Big_Mountain_Ticket_Price_From_The_Model)
#   * [5.8 Big Mountain Resort In Market Context](#5.8_Big_Mountain_Resort_In_Market_Context)
#     * [5.8.1 Ticket price](#5.8.1_Ticket_price)
#     * [5.8.2 Vertical drop](#5.8.2_Vertical_drop)
#     * [5.8.3 Snow making area](#5.8.3_Snow_making_area)
#     * [5.8.4 Total number of chairs](#5.8.4_Total_number_of_chairs)
#     * [5.8.5 Fast quads](#5.8.5_Fast_quads)
#     * [5.8.6 Runs](#5.8.6_Runs)
#     * [5.8.7 Longest run](#5.8.7_Longest_run)
#     * [5.8.8 Trams](#5.8.8_Trams)
#     * [5.8.9 Skiable terrain area](#5.8.9_Skiable_terrain_area)
#   * [5.9 Modeling scenarios](#5.9_Modeling_scenarios)
#     * [5.9.1 Scenario 1](#5.9.1_Scenario_1)
#     * [5.9.2 Scenario 2](#5.9.2_Scenario_2)
#     * [5.9.3 Scenario 3](#5.9.3_Scenario_3)
#     * [5.9.4 Scenario 4](#5.9.4_Scenario_4)
#   * [5.10 Summary](#5.10_Summary)
#   * [5.11 Further work](#5.11_Further_work)
# 

# ## 5.2 Introduction<a id='5.2_Introduction'></a>

# In this notebook, we now take our model for ski resort ticket price and leverage it to gain some insights into what price Big Mountain's facilities might actually support as well as explore the sensitivity of changes to various resort parameters. Note that this relies on the implicit assumption that all other resorts are largely setting prices based on how much people value certain facilities. Essentially this assumes prices are set by a free market.
# 
# We can now use our model to gain insight into what Big Mountain's ideal ticket price could/should be, and how that might change under various scenarios.

# ## 5.3 Imports<a id='5.3_Imports'></a>

# In[8]:


import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import cross_validate


# ## 5.4 Load Model<a id='5.4_Load_Model'></a>

# In[9]:


# This isn't exactly production-grade, but a quick check for development
# These checks can save some head-scratching in development when moving from
# one python environment to another, for example
expected_model_version = '1.0'
model_path = '../models/ski_resort_pricing_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if model.version != expected_model_version:
        print("Expected model version doesn't match version loaded")
    if model.sklearn_version != sklearn_version:
        print("Warning: model created under different sklearn version")
else:
    print("Expected model not found")


# ## 5.5 Load Data<a id='5.5_Load_Data'></a>

# In[10]:


ski_data = pd.read_csv('../data/ski_data_step3_features.csv')


# In[ ]:





# In[11]:


big_mountain = ski_data[ski_data.Name == 'Big Mountain Resort']


# In[12]:


big_mountain.T


# ## 5.6 Refit Model On All Available Data (excluding Big Mountain)<a id='5.6_Refit_Model_On_All_Available_Data_(excluding_Big_Mountain)'></a>

# This next step requires some careful thought. We want to refit the model using all available data. But should we include Big Mountain data? On the one hand, we are _not_ trying to estimate model performance on a previously unseen data sample, so theoretically including Big Mountain data should be fine. One might first think that including Big Mountain in the model training would, if anything, improve model performance in predicting Big Mountain's ticket price. But here's where our business context comes in. The motivation for this entire project is based on the sense that Big Mountain needs to adjust its pricing. One way to phrase this problem: we want to train a model to predict Big Mountain's ticket price based on data from _all the other_ resorts! We don't want Big Mountain's current price to bias this. We want to calculate a price based only on its competitors.

# In[13]:


X = ski_data.loc[ski_data.Name != "Big Mountain Resort", model.X_columns]
y = ski_data.loc[ski_data.Name != "Big Mountain Resort", 'AdultWeekend']


# In[14]:


len(X), len(y)


# In[15]:


model.fit(X, y)


# In[16]:


cv_results = cross_validate(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)


# In[17]:


cv_results['test_score']


# In[18]:


mae_mean, mae_std = np.mean(-1 * cv_results['test_score']), np.std(-1 * cv_results['test_score'])
mae_mean, mae_std


# These numbers will inevitably be different to those in the previous step that used a different training data set. They should, however, be consistent. It's important to appreciate that estimates of model performance are subject to the noise and uncertainty of data!

# ## 5.7 Calculate Expected Big Mountain Ticket Price From The Model<a id='5.7_Calculate_Expected_Big_Mountain_Ticket_Price_From_The_Model'></a>

# In[19]:


X_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", model.X_columns]
y_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", 'AdultWeekend']


# In[20]:


bm_pred = model.predict(X_bm).item()


# In[21]:


y_bm = y_bm.values.item()


# In[22]:


print(f'Big Mountain Resort modelled price is ${bm_pred:.2f}, actual price is ${y_bm:.2f}.')
print(f'Even with the expected mean absolute error of ${mae_mean:.2f}, this suggests there is room for an increase.')


# This result should be looked at optimistically and doubtfully! The validity of our model lies in the assumption that other resorts accurately set their prices according to what the market (the ticket-buying public) supports. The fact that our resort seems to be charging that much less that what's predicted suggests our resort might be undercharging. 
# But if ours is mispricing itself, are others? It's reasonable to expect that some resorts will be "overpriced" and some "underpriced." Or if resorts are pretty good at pricing strategies, it could be that our model is simply lacking some key data? Certainly we know nothing about operating costs, for example, and they would surely help.

# ## 5.8 Big Mountain Resort In Market Context<a id='5.8_Big_Mountain_Resort_In_Market_Context'></a>

# Features that came up as important in the modeling (not just our final, random forest model) included:
# * vertical_drop
# * Snow Making_ac
# * total_chairs
# * fastQuads
# * Runs
# * LongestRun_mi
# * trams
# * SkiableTerrain_ac

# A handy glossary of skiing terms can be found on the [ski.com](https://www.ski.com/ski-glossary) site. Some potentially relevant contextual information is that vertical drop, although nominally the height difference from the summit to the base, is generally taken from the highest [_lift-served_](http://verticalfeet.com/) point.

# It's often useful to define custom functions for visualizing data in meaningful ways. The function below takes a feature name as an input and plots a histogram of the values of that feature. It then marks where Big Mountain sits in the distribution by marking Big Mountain's value with a vertical line using `matplotlib`'s [axvline](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.axvline.html) function. It also performs a little cleaning up of missing values and adds descriptive labels and a title.

# In[23]:


#Code task 1#
#Add code to the `plot_compare` function that displays a vertical, dashed line
#on the histogram to indicate Big Mountain's position in the distribution
#Hint: plt.axvline() plots a vertical line, its position for 'feature1'
#would be `big_mountain['feature1'].values, we'd like a red line, which can be
#specified with c='r', a dashed linestyle is produced by ls='--',
#and it's nice to give it a slightly reduced alpha value, such as 0.8.
#Don't forget to give it a useful label (e.g. 'Big Mountain') so it's listed
#in the legend.
def plot_compare(feat_name, description, state=None, figsize=(10, 5)):
    """Graphically compare distributions of features.
    
    Plot histogram of values for all resorts and reference line to mark
    Big Mountain's position.
    
    Arguments:
    feat_name - the feature column name in the data
    description - text description of the feature
    state - select a specific state (None for all states)
    figsize - (optional) figure size
    """
    
    plt.subplots(figsize=figsize)
    # quirk that hist sometimes objects to NaNs, sometimes doesn't
    # filtering only for finite values tidies this up
    if state is None:
        ski_x = ski_data[feat_name]
    else:
        ski_x = ski_data.loc[ski_data.state == state, feat_name]
    ski_x = ski_x[np.isfinite(ski_x)]
    plt.hist(ski_x, bins=30)
    plt.axvline(x=big_mountain[feat_name].values, c='r', ls='--', alpha=0.8, label='Big Mountain')
    plt.xlabel(description)
    plt.ylabel('frequency')
    plt.title(description + ' distribution for resorts in market share')
    plt.legend()


# ### 5.8.1 Ticket price<a id='5.8.1_Ticket_price'></a>

# Look at where Big Mountain sits overall amongst all resorts for price and for just other resorts in Montana.

# In[24]:


plot_compare('AdultWeekend', 'Adult weekend ticket price ($)')


# In[25]:


plot_compare('AdultWeekend', 'Adult weekend ticket price ($) - Montana only', state='Montana')


# ### 5.8.2 Vertical drop<a id='5.8.2_Vertical_drop'></a>

# In[26]:


plot_compare('vertical_drop', 'Vertical drop (feet)')


# Big Mountain is doing well for vertical drop, but there are still quite a few resorts with a greater drop.

# ### 5.8.3 Snow making area<a id='5.8.3_Snow_making_area'></a>

# In[27]:


plot_compare('Snow Making_ac', 'Area covered by snow makers (acres)')


# Big Mountain is very high up the league table of snow making area.

# ### 5.8.4 Total number of chairs<a id='5.8.4_Total_number_of_chairs'></a>

# In[28]:


plot_compare('total_chairs', 'Total number of chairs')


# Big Mountain has amongst the highest number of total chairs, resorts with more appear to be outliers.

# ### 5.8.5 Fast quads<a id='5.8.5_Fast_quads'></a>

# In[29]:


plot_compare('fastQuads', 'Number of fast quads')


# Most resorts have no fast quads. Big Mountain has 3, which puts it high up that league table. There are some values  much higher, but they are rare.

# ### 5.8.6 Runs<a id='5.8.6_Runs'></a>

# In[30]:


plot_compare('Runs', 'Total number of runs')


# Big Mountain compares well for the number of runs. There are some resorts with more, but not many.

# ### 5.8.7 Longest run<a id='5.8.7_Longest_run'></a>

# In[31]:


plot_compare('LongestRun_mi', 'Longest run length (miles)')


# Big Mountain has one of the longest runs. Although it is just over half the length of the longest, the longer ones are rare.

# ### 5.8.8 Trams<a id='5.8.8_Trams'></a>

# In[32]:


plot_compare('trams', 'Number of trams')


# The vast majority of resorts, such as Big Mountain, have no trams.

# ### 5.8.9 Skiable terrain area<a id='5.8.9_Skiable_terrain_area'></a>

# In[33]:


plot_compare('SkiableTerrain_ac', 'Skiable terrain area (acres)')


# Big Mountain is amongst the resorts with the largest amount of skiable terrain.

# ## 5.9 Modeling scenarios<a id='5.9_Modeling_scenarios'></a>

# In[34]:


expected_visitors = 350_000


# In[35]:


all_feats = ['vertical_drop', 'Snow Making_ac', 'total_chairs', 'fastQuads', 
             'Runs', 'LongestRun_mi', 'trams', 'SkiableTerrain_ac']
big_mountain[all_feats]


# In[36]:


#Code task 2#
#In this function, copy the Big Mountain data into a new data frame
#(Note we use .copy()!)
#And then for each feature, and each of its deltas (changes from the original),
#create the modified scenario dataframe (bm2) and make a ticket price prediction
#for it. The difference between the scenario's prediction and the current
#prediction is then calculated and returned.
#Complete the code to increment each feature by the associated delta
def predict_increase(features, deltas):
    """Increase in modelled ticket price by applying delta to feature.
    
    Arguments:
    features - list, names of the features in the ski_data dataframe to change
    deltas - list, the amounts by which to increase the values of the features
    
    Outputs:
    Amount of increase in the predicted ticket price
    """
    
    bm2 = X_bm.copy()
    for f, d in zip(features, deltas):
        bm2[f] += d
    return model.predict(bm2).item() - model.predict(X_bm).item()


# ### 5.9.1 Scenario 1<a id='5.9.1_Scenario_1'></a>

# Close up to 10 of the least used runs. The number of runs is the only parameter varying.

# In[37]:


[i for i in range(-1, -11, -1)]


# In[38]:


runs_delta = [i for i in range(-1, -11, -1)]
price_deltas = [predict_increase(['Runs'], [delta]) for delta in runs_delta]


# In[39]:


price_deltas


# In[40]:


#Code task 3#
#Create two plots, side by side, for the predicted ticket price change (delta) for each
#condition (number of runs closed) in the scenario and the associated predicted revenue
#change on the assumption that each of the expected visitors buys 5 tickets
#There are two things to do here:
#1 - use a list comprehension to create a list of the number of runs closed from `runs_delta`
#2 - use a list comprehension to create a list of predicted revenue changes from `price_deltas`
runs_closed = [-1 * run for run in runs_delta] #1
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.5)
ax[0].plot(runs_closed, price_deltas, 'o-')
ax[0].set(xlabel='Runs closed', ylabel='Change ($)', title='Ticket price')
revenue_deltas = [5 * expected_visitors * price for price in price_deltas] #2
ax[1].plot(runs_closed, revenue_deltas, 'o-')
ax[1].set(xlabel='Runs closed', ylabel='Change ($)', title='Revenue');


# The model says closing one run makes no difference. Closing 2 and 3 successively reduces support for ticket price and so revenue. If Big Mountain closes down 3 runs, it seems they may as well close down 4 or 5 as there's no further loss in ticket price. Increasing the closures down to 6 or more leads to a large drop. 

# ### 5.9.2 Scenario 2<a id='5.9.2_Scenario_2'></a>

# In this scenario, Big Mountain is adding a run, increasing the vertical drop by 150 feet, and installing an additional chair lift.

# In[41]:


#Code task 4#
#Call `predict_increase` with a list of the features 'Runs', 'vertical_drop', and 'total_chairs'
#and associated deltas of 1, 150, and 1
ticket2_increase = predict_increase(['Runs', 'vertical_drop', 'total_chairs'], [1, 150, 1])
revenue2_increase = 5 * expected_visitors * ticket2_increase


# In[42]:


print(f'This scenario increases support for ticket price by ${ticket2_increase:.2f}')
print(f'Over the season, this could be expected to amount to ${revenue2_increase:.0f}')


# ### 5.9.3 Scenario 3<a id='5.9.3_Scenario_3'></a>

# In this scenario, you are repeating the previous one but adding 2 acres of snow making.

# In[43]:


#Code task 5#
#Repeat scenario 2 conditions, but add an increase of 2 to `Snow Making_ac`
ticket3_increase = predict_increase(['Runs', 'vertical_drop', 'total_chairs', 'Snow Making_ac'], [1, 150, 1, 2])
revenue3_increase = 5 * expected_visitors * ticket3_increase


# In[44]:


print(f'This scenario increases support for ticket price by ${ticket3_increase:.2f}')
print(f'Over the season, this could be expected to amount to ${revenue3_increase:.0f}')


# Such a small increase in the snow making area makes no difference!

# ### 5.9.4 Scenario 4<a id='5.9.4_Scenario_4'></a>

# This scenario calls for increasing the longest run by .2 miles and guaranteeing its snow coverage by adding 4 acres of snow making capability.

# In[45]:


#Code task 6#
#Predict the increase from adding 0.2 miles to `LongestRun_mi` and 4 to `Snow Making_ac`
predict_increase(['LongestRun_mi', 'Snow Making_ac'], [.2,4])


# No difference whatsoever. Although the longest run feature was used in the linear model, the random forest model (the one we chose because of its better performance) only has longest run way down in the feature importance list. 

# ## 5.10 Summary<a id='5.10_Summary'></a>

# **Q: 1** Write a summary of the results of modeling these scenarios. Start by starting the current position; how much does Big Mountain currently charge? What does your modelling suggest for a ticket price that could be supported in the marketplace by Big Mountain's facilities? How would you approach suggesting such a change to the business leadership? Discuss the additional operating cost of the new chair lift per ticket (on the basis of each visitor on average buying 5 day tickets) in the context of raising prices to cover this. For future improvements, state which, if any, of the modeled scenarios you'd recommend for further consideration. Suggest how the business might test, and progress, with any run closures.

# **A: 1**Big Mountain resort currently has a ticket price of $81.00. Our modeling suggests that a ticket price of $94.49 can be supported by current facilities. 
# 
# 
# According to our model, the resort can close up-to 5 runs without significant negative affect to ticket price support and revenue. 
# 
# Adding one chair, one run and 150 vertical feet increases support for ticket price by $2.19, which leads to increased revenue of $3.83 million over the course of a season. (given the assumption of 350,000 customers buying 5 tickets). Operating cost for the new chairlift is given at $1.43 million. Thus, the projected net profit is $2.29 million
# 2
# For future improvement, I recommend additional modeling on the value add of additional fastQuad chairlifts. Replacing an old double or triple chairlift with a highspeed quad would increase ticket price support, however the high maintenance and operating cost of these chairlifts raise concerns over sustained profitability.
# 
# I recommend that in the upcoming ski season, the company tracks number of tickets sold per day based on the number of open runs. If our thesis holds, (the resort can close up-to 5 runs without significant negative affect to ticket price support and revenue), proposals to permanently close 5 runs with low marginal utility should be presented.  
# 

# ## 5.11 Further work<a id='5.11_Further_work'></a>

# **Q: 2** What next? Highlight any deficiencies in the data that hampered or limited this work. The only price data in our dataset were ticket prices. You were provided with information about the additional operating cost of the new chair lift, but what other cost information would be useful? Big Mountain was already fairly high on some of the league charts of facilities offered, but why was its modeled price so much higher than its current price? Would this mismatch come as a surprise to the business executives? How would you find out? Assuming the business leaders felt this model was useful, how would the business make use of it? Would you expect them to come to you every time they wanted to test a new combination of parameters in a scenario? We hope you would have better things to do, so how might this model be made available for business analysts to use and explore?

# **A: 2** The next step will be to add to our dataset with more quantitative data from the alpine resort industry like prices for hotels, equipment rentals, and ski or snowboard lesson. Operating cost information on highspeed quads, snowmaking will allow the data analyst to present the board with more recommendations on augmentations to the business model to increase profitability.  
# 
# As presented 
# The modeled price being higher than the current price can be understood view that rational given in Section 5.7 above:
# “This result should be looked at optimistically and doubtfully! The validity of our model lies in the assumption that other resorts accurately set their prices according to what the market (the ticket-buying public) supports. The fact that our resort seems to be charging that much less that what's predicted suggests our resort might be undercharging. But if ours is mispricing itself, are others? It's reasonable to expect that some resorts will be "overpriced" and some "underpriced."
# 
# Additionally, variations in purchasing power between US regions needs to be considered. Our resort my be underpriced in comparisons to prestige resorts like Vail and Jackson Hole that serve ultra high net worth individuals. However Big Mountain is the most expensive resort in the State of Montana. The Average ticket price in Montana is $51.9 dollars. In fact, only two other mountains Red Lodge Mountain and Bridger Bowl have a ticket price over 60 dollars. 
# 
# This model can be used as the backend of a Tableu or Power Bi dashboard to allow business executives to test new combinations of parameters. 
# 
# 
