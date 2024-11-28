# ML-Project-2

>>1ST MAIL OF THE PROF
> Each file is in the csv format and shows the x-y center of mass coordinates of one worm.
>
> Under "Lifespan" you will find "companyDrug" and "control" subfolders, which are from one experiment where a drug turned out not to have an effect on lifespan. However, can you find a difference in behavior in the presence of the drug? Note that the lifespan recordings were made for 6 hours at a time and the csv file thus restarts the frame numbering after each 10800 frames.
>
> More lifespan data to come.
>
> Under "Optogenetics" you will find the "ATR+" and "ATR-" subfolders. ATR+ means that the optogenetic systems in the worm neurons were functional. ATR- is the control. Here, the csv files have an additional column "Light_Pulse", which is 1 when the light was turned on. Can you find worm personalities, i.e., persistent differences between worms, including in response to the light stimulus? Here is a related reference


>>>INFORMATIONS FROM SLACK LPBS
5) The data was recorded at a constant interval of 2 seconds over 30 minutes every 6 hours, so there should be 900 frames per session.


>>> WHAT I ANALYSED (might be wrong)
Key Takeaways from our Data 

Each frame corresponds to 2 seconds of elapsed time.
A session lasts 30 minutes = 900 frames = 1800 seconds
After each session, there is a gap of 5.5 hours, during which no data is recorded.

The Frame column resets to 1 after 10'800 frames.
This means Frame alone does not provide a continuous timeline for the worm’s behavior.

So for a session + the gap it's 6 hours. So 900 frames correspond to 30min recording + 5.5 h of gap = 6 hours
In 10'800 frames we have 12 sessions of 900 frames. So before the frames resets it's 12 * 0.5h + 12 * 5.5h = 72h so 3 days


>> Handling NaN values :
I don't get it why it must be a bug with the microscope yes, but sometimes the values are just not given. for a few frames. But for a particular frame, if value of x is not given, it's neither for y and the speed thus equals 0 (must be in the skeleton code that if missing position --> set the speed to 0.