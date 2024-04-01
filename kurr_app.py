from functions import create_agents
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from recipe_db import cuisines
import plotly.graph_objects as go
import altair as alt
import time
from sklearn.cluster import KMeans
from Final import simulation
from matplotlib.patches import ConnectionPatch
from sklearn import tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas_profiling import ProfileReport
import streamlit.components.v1 as components
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import numpy as np


@st.cache(suppress_st_warning=True)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def get_chart(data):
    hover = alt.selection_single(
        fields=["RunID"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
    lines = (
        alt.Chart(data, title="Number of Right-swipes per iteration")
        .mark_line()
        .encode(
            x="RunID",
            y="Choice",
            color="Agent_diet",
        )
    )
    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)
    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="RunID",
            y="Choice",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("RunID", title="Iteration"),
                alt.Tooltip("Choice", title="Number of right swipes"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()


def matches(df):
    x = (df.apply(lambda row: row['Recipe_cuisine'] in row['Agent_cuisine'], axis='columns'))
    return x.astype(int)


app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Simulation', 'Analysis'])
############################################# HOME PAGE ################################################################
if app_mode == "Home":
    st.title("An agent-based model of user interaction on the app 'Kurr'")
    st.title("Model synopsis")
    st.subheader("Step 1: Generate users")
    st.markdown("**Dietary preferences**")
    st.write("A user-specified number of agents are generated with different dietary preferences.")
    st.markdown("The dietary preferences (or filters) are *vegan*, *vegetarian*, and *omnivore*.")
    st.write(
        "Based on these dietary preferences, the agents interact differently with the recipes on the app. Vegans "
        "interact only with vegan recipes,vegetarians interact with both vegan and vegetarian recipes, while omnivores "
        "interact with all of them")
    st.markdown("**Cuisine preferences**")
    st.write(
        "Each agent also have a random number of cuisine preferences. The cuisine preferences are based on the recipe "
        "database provided by 'Kurr'.")
    st.write(
        "In total there are 18 different cuisines. For each agent we draw a random number (with equal probability) "
        "between 1 and 18. This number is then used to decide how many different cuisines the agent prefers.")
    st.write(
        "For example, if the result of the random number generator is 3, we sample 3 cuisines from the full list of "
        "cuisines and assign those to the agent. Resulting in the following structure:")
    df = pd.DataFrame({"User": "User-1", "Cuisines": ['Thai,Indian,Husmanskost'], "Filter": "Vegan"})
    fig = go.Figure(data=[go.Table(header=dict(values=['User', 'Cuisine', "Filter"]),
                                   cells=dict(values=["User-1", 'Thai,Indian,Husmanskost', "Vegan"]))
                          ])
    st.plotly_chart(fig)

    st.subheader("Step 2: Simulating interaction")
    st.markdown("**Steps**")
    st.write(
        "At each step of the simulation, the agents are presented with a recipe and decides to either 'swipe left' or "
        "'swipe right' on the recipe. The likelihood of 'swiping right' is determined by two factors.")
    st.markdown(
        "- **Base probability:** Each agent are assigned a vector of probabilities that represent the likelihood of a "
        "left-or right swipe. The probability vector is constant in each of the agent types. Meaning, all vegans have "
        "the same base probability of liking a recipe")
    st.markdown(
        "- **Conditional probability:** As mentioned earlier, the recipe might also match with the agents preferences "
        "of cuisine. If it does match then the agents have a higher probability to 'swipe right' on the recipe than if "
        "there were no match")
    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)
    st.write("To exemplify, look at the following table of probabilities:")
    st.latex(r'''
    \begin{matrix}
    & Left & Right \\
    Base & 0.5 & 0.5 \\
    Conditional & 0.2 & 0.8
    \end{matrix}''')
    st.write(
        "In the base-probability case (corresponding to no match between agent cuisine and recipe cuisine), the probability to swipe left or right is 0.5 respectively. In the conditional probability case (corresponding to a match between recipe cuisine and agent cuisine preference), the probability to swipe right is higher. We implemented this feature to create a more realistic model. The probabilities can be changed using the probability sliders in the sidebar")
    st.write(
        "To add further complexity to the model, the base -and conditional probabilities are different depending on the dietary preferences. For instance, a vegetarian most likely eats vegan food as well but prefers vegetarian because s/he likes dairy products. To capture such phenomena we added another vector of probabilities for vegetarians. The first vector represents the probabilities of left/right swipe on vegan recipes and the second vector represent the same probabilities but for vegetarian recipes.The same procedure was implemented for omnivores as well.")
    st.write("The probabilities of liking a given recipe for all the agent types are represented in the matrix below:")
    st.latex(r'''
        \begin{matrix}
        & & Vegan & & Vegetarian & & Omnivore &\\
        & &Left & Right & Left & Right & Left & Right\\
        Vegan & Base & 0.5 & 0.5 & & & &\\
        & Conditional & 0.3 & 0.7 & & & &\\
        Vegetarian & Base & 0.6 & 0.4 & 0.4 & 0.6 & &\\
        & Conditional & 0.5 & 0.5 & 0.3 & 0.7 & &\\
        Omnivore & Base & 0.8 & 0.2 & 0.7 & 0.3 & 0.6 & 0.4\\
        & Conditional & 0.7 & 0.3 & 0.6 & 0.4 & 0.1 & 0.9
        \end{matrix}''')
    st.title("Interacting with the model")
    st.subheader("Simulation page")
    st.markdown(
        "**Step 1:** Specify model parameters. Use the sliders in the left sidebar to specify the number of iterations and agents.")
    st.markdown(
        "**Step 2:** Hit the 'Generate agents' button. This will generate the agents and some descriptive statistics and plots of the population")
    st.markdown(
        "**Step 3:** Hit the 'Run simulation' button. This will start the simulation and end at the specified maximum number of steps")
    st.write(
        "When the simulation is finished, the user may choose to run prediction and visualization models by changing the 'Selected Page' to analysis")
    st.subheader("Analysis page")
    st.write(
        "The analysis page provides the users with the options of running simple Explanatory Data Analysis (EDA), as well as more sophisticated prediction/clustering models.")
    st.write(
        "At the bottom of the EDA section, the user may choose to download the dataset resulting from the simulation.")
    st.write(
        "If the user choose to clustering, a Kmeans algorithm will be executed with the user specified arguments. Heuristical tools of how many clusters to choose are presented together with a table displaying the proportion of matching recipes (cuisines) and proportion of left swipes in each cluster.")
    st.write(
        "If Decision Tree Analysis is selected a decision tree algorithm will be executed with the user-specified arguments. When the algorithm has converged, model statistics are shown together with a visualization of the tree.")

############################################# SIMULATION PAGE ###############################################################
if 'result' not in st.session_state:
    st.session_state.result = None

if app_mode == "Simulation":

    st.sidebar.title("Model specifications")
    st.sidebar.subheader("Number of Steps (Iterations)")
    n = st.sidebar.slider("N", 0, 500, 100, step=50)
    st.sidebar.subheader("Number of agents")
    n_vegan = st.sidebar.slider("Vegan", 0, 1000, 100, step=50)
    n_vegetarian = st.sidebar.slider("Vegetarian", 0, 1000, 100, step=50)
    n_omnivore = st.sidebar.slider("Omnivore", 0, 1000, 100, step=50)
    # st.sidebar.button("Generate users")
    agents = create_agents(n_vegan, n_vegetarian, n_omnivore)
    sim = simulation(agents, n)

    if st.sidebar.button("Generate users"):
        # CREATING AGENTS

        # GETTING THEIR DIETARY PREFERENCES
        cuisines3 = []
        for agent in agents:
            cuisines3.append(agent.cuisine)
        res = []
        for cui in cuisines:
            res.append([cui, sum(x.count(cui) for x in cuisines3)])

        categories = []
        count = []
        for i in range(len(res)):
            categories.append(res[i][0])
            count.append(res[i][1])
        ##################################################################
        # PLOTTING
        fig, ax = plt.subplots()

        bar_labels = ['red', 'blue', '_red', 'orange']
        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
        ax.bar(categories, count, )
        ax.set_title("Agent cuisine preferences")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        progress_bar = st.progress(0)
        for i in range(n):
            time.sleep(0.01)
            progress_bar.progress((i + 1) / n)
        st.success('Users generated!', icon="✅")

    if st.sidebar.button("Run simulation"):
        try:
            sim = simulation(agents, n)
            progress_bar = st.progress(0)
            for i in range(n):
                time.sleep(0.01)
                progress_bar.progress((i + 1) / n)
            st.success('Simulation complete!', icon="✅")
            results = sim
        finally:
            st.session_state.result = results

        # VISUALIZATION

        match = matches(results)

        results["Match"] = match
        # st.write(results)
        st.title("Descriptive statistics")
        data_csv = convert_df(results)
        st.download_button(label="Download Data", data=data_csv, file_name="Simulation-results.csv")

        groups = ["Vegan", "Vegetarian", "Omnivore"]

        # newdf2 = results.groupby(['Agent_diet'])['Choice'].mean()
        x1 = results[results["Agent_diet"] == "vegan"]["Choice"]
        x2 = results[results["Agent_diet"] == "vegetarian"]["Choice"]
        x3 = results[results["Agent_diet"] == "omnivore"]["Choice"]

        pie = pd.Series((x1.mean(), x2.mean(), x3.mean()))

        ############################################################################################################
        # make figure and assign axis objects
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(9, 5))
        fig2.subplots_adjust(wspace=0)
        x4 = [results["Choice"].mean(), (1 - results["Choice"].mean())]
        # pie chart parameters
        labels_x4 = ["Swipe right", "Swipe left"]
        explode = [0.1, 0]

        # rotate so that first wedge is split by the x-axis
        angle = -180 * x4[0]
        wedges, *_ = ax2.pie(x4, autopct='%1.1f%%', startangle=angle,
                             labels=labels_x4, explode=explode)
        # bar chart parameters
        bottom = sum(pie)
        width = .2

        # Adding from the top matches the legend.
        for j, (height, label) in enumerate(reversed([*zip(pie, groups)])):
            bottom -= height
            bc = ax3.bar(0, height, width, bottom=bottom, color='C0', label=label,
                         alpha=0.1 + 0.25 * j)
            ax3.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        ax3.set_title('Proportion based on diet')
        ax3.legend()
        ax3.axis('off')
        ax3.set_xlim(- 2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(pie)

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax3.transData,
                              xyB=(x, y), coordsB=ax2.transData)

        con.set_color([0, 0, 0])
        con.set_linewidth(2)
        ax3.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax3.transData,
                              xyB=(x, y), coordsB=ax2.transData)
        con.set_color([0, 0, 0])
        ax3.add_artist(con)
        con.set_linewidth(2)
        st.pyplot(fig2)

        ############################################################################################################
        rec_diet_df = results.groupby(["Agent_diet", "Recipe_diet", "Match"], as_index=False)['Choice'].sum()
        rec_diet_df["Match"] = np.where(rec_diet_df["Match"] == 0, "No match", "Match")

        bar_chart = alt.Chart(rec_diet_df).mark_bar().encode(
            x=alt.X("Match:N", title=None),
            y=alt.Y("Choice:Q", title="Number of 'right-swipes'"),
            color="Recipe_diet:N",
            column=alt.Column("Agent_diet:O", title="Agent diet"),
            tooltip=[
                alt.Tooltip("Match", title="Matching recipe"),
                alt.Tooltip("Recipe_diet", title="Recipe diet"),
                alt.Tooltip("Choice", title="Number of right swipes"),
            ]

        ).properties(title="CHICKIIIES", width=160, height=100)

        st.altair_chart(bar_chart)
        ############################################################################################################
        # Reformatting the data
        results['RunID'] = results['RunID'].astype(int)
        df1 = results.groupby(['RunID', "Agent_diet"], as_index=False)['Choice'].sum()
        # st.line_chart(time_data)
        ############################################################################################################

        chart = get_chart(df1)
        st.altair_chart(chart, use_container_width=True)
        st.write(results)
columns = ["Match", "Recipe_cuisine", "Time", "Choice"]

###################################################################################################################
############################################ ANALYSIS PAGE ########################################################
###################################################################################################################


if app_mode == "Analysis":

    st.markdown("<h1 style='text-align: center; color: #ff0000;'>Analysis</h1>", unsafe_allow_html=True)
    mode = st.sidebar.radio("Mode", ["EDA", "Prediction"])
    if mode == "EDA":
        st.header("Exploratory Data Analysis - EDA")
        data_csv = convert_df(st.session_state.result)
        profile = ProfileReport(st.session_state.result)
        profile.to_file("Kurr.html")
        HtmlFile = open("Kurr.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()
        print(source_code)
        components.html(source_code, height=1080, scrolling=True)
        st.download_button(label="Download Data", data=data_csv, file_name="Simulation_Results_Data.csv")

    if mode == "Prediction":
        ################################################################################################################
        # WRANGLING DATA
        data = st.session_state.result
        data = data.assign(Time_cat=pd.cut(data["Time"],
                                           bins=[0, 30, 60, 999],
                                           labels=["Low", "Medium", "High"]))
        data_x = data.drop(
            ["Agent_cuisine", "RunID", "Recipe_cuisine", "User-ID", "Recipe_diet", "Choice", "Recipe", "Time"], axis=1)
        data_x_encoded = pd.get_dummies(data_x, drop_first=True)
        data_y = data["Choice"]

        df = data
        df = df.sample(n=1000)
        # Handling numerical features
        numerical_features = ["Time"]
        scaler = StandardScaler()
        scaler.fit(df[numerical_features])
        numerical_data = scaler.transform(df[numerical_features])
        numerical_data = pd.DataFrame(numerical_data, index=df.index, columns=numerical_features)
        ################################################################################################################
        # Handling ordinal features
        # create some lists
        ordinal_features = ['Agent_diet']
        df['Agent_diet'] = df['Agent_diet'].map({"vegan": 0, "vegetarian": 1, "omnivore": 2})
        ordinal_data = df[ordinal_features]

        # MinMax scaled
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[ordinal_features])
        ordinal_data = scaler.transform(df[ordinal_features])
        ordinal_data = pd.DataFrame(ordinal_data, index=df.index, columns=ordinal_features)
        ################################################################################################################
        categorical_features = ["Match", "Choice"]
        nominal_features = [c for c in categorical_features if c not in ordinal_features]

        nominal_data = list()
        for i, x in df[nominal_features].nunique().iteritems():
            if x <= 2:
                nominal_data.append(pd.get_dummies(df[[i]], drop_first=True))
            elif x > 2:
                nominal_data.append(pd.get_dummies(df[[i]], drop_first=False))

        nominal_data = pd.concat(nominal_data, axis=1)
        ################################################################################################################
        ######################################## CLUSTERING VISUALIZATION ##############################################
        ################################################################################################################
        # transformed and scaled dataset
        from sklearn.decomposition import PCA

        Xy_scaled = pd.concat([numerical_data, nominal_data, ordinal_data], axis=1)
        # Xy_scaled = Xy_scaled.sample(n=5000)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(Xy_scaled)
        feat = list(range(pca.n_components_))
        PCA_components = pd.DataFrame(principalComponents, columns=list(range(len(feat))))
        choosed_component = [0, 1]

        ################################################################################################################
        Model = st.sidebar.selectbox('Select Model/Algorithm',
                                     ['Decision Tree Classifier',
                                      'KMeans Clustering'])
        k = []


        def hyperparameters(Model):
            params = {}
            if Model == "Decision Tree Classifier":
                params = {}
                max = st.sidebar.slider("Max depth", min_value=1, max_value=50, value=1)
                min_samples_leaf = st.sidebar.slider("Min samples per leaf", min_value=1, max_value=1000, value=1)
                min_samples_split = st.sidebar.slider("Min samples per split", min_value=1, max_value=1000, value=1)
                random_state = st.sidebar.text_input("Random State", value=0)
                params["max_depth"] = max
                params["min_samples_split"] = min_samples_split
                params["min_samples_leaf"] = min_samples_leaf
                params["random_state"] = int(random_state)
            elif Model == 'KMeans Clustering':
                K = st.sidebar.slider("Number of Clusters", 1, 10)
                k.append(K)
                params["n_clusters"] = int(K)
            return params


        params = hyperparameters(Model)
        st.sidebar.markdown("Scroll Below in Main window to make your own predictions.")


        def classifier(Model, params):
            if Model == "Decision Tree Classifier":
                st.header("Decision Tree Classifier")
                clf = DecisionTreeClassifier(**params)
            elif Model == 'KMeans Clustering':
                st.header('KMeans Clustering')
                clf = KMeans(**params)
            return clf


        clf = classifier(Model, params)
        x_train, x_test, y_train, y_test = train_test_split(data_x_encoded, data_y, test_size=0.2)
        if Model == 'KMeans Clustering':
            clf.fit(Xy_scaled)
        else:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            st.header("Evaluation Metrics")
            acc = accuracy_score(y_test, y_pred)
            st.subheader(f"Accuracy Score :")
            st.write(acc)
            prec = precision_score(y_test, y_pred, average=None)
            st.subheader(f"Precision Score : ")
            st.write(prec)
            st.subheader("Classification Report :")
            report = classification_report(y_test, y_pred, output_dict=True, target_names=["Left-swipe", "Right-swipe"])
            st.dataframe(report)


        def visualizations(Model):
            table_viz = ["Match", "Choice", "Time"]
            eda = data.groupby("Agent_diet").agg("mean")[table_viz]
            # st.dataframe(eda)

            if Model == 'KMeans Clustering':
                results = dict()

                for i in range(2, 11):
                    X = PCA_components[choosed_component]
                    kmeans = KMeans(n_clusters=i).fit(X)
                    score0 = kmeans.inertia_
                    y_kmeans = kmeans.predict(X)
                    df["cluster"] = y_kmeans
                    score1 = silhouette_score(X, kmeans.labels_, metric='euclidean')
                    score2 = silhouette_score(X, kmeans.labels_, metric='correlation')
                    results[i] = {'k': kmeans, 's0': score0, 's1': score1, 's2': score2}

                fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 3))
                axs[0].plot([i for i in results.keys()], [i['s0'] for i in results.values()], 'o-', label='Inertia')
                axs[1].plot([i for i in results.keys()], [i['s1'] for i in results.values()], 'o-', label='Euclidean')
                axs[1].plot([i for i in results.keys()], [i['s2'] for i in results.values()], 'o-', label='Correlation')

                for ax in axs:
                    ax.set_xticks(range(2, 11))
                    ax.set_xlabel('K')
                    ax.legend()
                st.pyplot(fig)
                n_k = 0

                for i in k:
                    n_k = i
                index = [f"Cluster {i + 1}" for i in range(n_k)]

                viz_df = df[df["cluster"] <= n_k - 1]
                eda = viz_df.groupby("cluster").agg(["mean"])[table_viz]
                eda2 = viz_df.groupby("cluster").count()["Choice"]
                eda.index = index

                # newdf2 = results.groupby(['Agent_diet'])['Choice'].mean()

                choice_pie = []
                labels = []
                for i in range(n_k):
                    x = viz_df[viz_df["cluster"] == i]["Choice"].mean()
                    labels.append(f"Cluster {i}")
                    choice_pie.append(x)

                pie = pd.Series(choice_pie)
                pie.index = index

                fig1, ax1 = plt.subplots()
                ax1.pie(pie, labels=labels, autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                st.table(eda)













            elif Model == 'Decision Tree Classifier':
                dot_data = tree.export_graphviz(clf, out_file=None,
                                                feature_names=["Recipe match",
                                                               "Vegan",
                                                               "Vegetarian",
                                                               "Cooking time:Medium",
                                                               "Cooking time:high"],
                                                class_names=["Left", "Right"],
                                                filled=True)
                st.graphviz_chart(dot_data, use_container_width=True)


        visualizations(Model)
