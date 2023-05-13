## Researching Student performance related to video-watching behavior:

`behavior-performance.txt` contains data for an online course on how students watched videos (e.g., how much time they spent watching, how often they paused the video, etc.) and how they performed on in-video quizzes. `readme.pdf` details the information contained in the data fields. The aim of this project is to use this data to perform analysis tests using various data science techniques and prediction algorithms. We plan to run prediction algorithms for all students for one video and repeat this process for all the videos.


1. The first question we aim to answer is how well can the students be naturally grouped or clustered by their video-watching behavior (`fracSpent`, `fracComp`, `fracPaused`, `numPauses`, `avgPBR`, `numRWs`, and `numFFs`)? We will use all students that complete at least five of the videos in our analysis.

2. Afterwards we plan to see if student's video-watching behavior can be used to predict a student's performance?

3. Taking this a step further, we will examine how well we can predict a student's performance on a *particular* in-video quiz question (i.e., whether they will be correct or incorrect) based on their video-watching behaviors while watching the corresponding video?