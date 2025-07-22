# STCell
The investigation of time cell and place cell unification in the hippocampus CA3 subfield.

## TODOs
- [x] Push code to GitHub.
- [x] Organize the code, try to keep the model training and the analyzing code separate.
- [x] **Very Important** Do a sanity test on Zhaoze's experiment.
- [ ] Add more references to the intro.
- [x] For figure 2 panel a,b,c. Add another (few) plots that indicate the reuse of time cells (hidden units) for the space task (similar to the current figure 2.b).
- [ ] For the ring plots, plot each ratemap's min, max, mean. (use np.nanmean/nanmin/nanmax)

- [x] *July 13*: add figure and 2 and 3 to Overleaf.
- [ ] I want to add one more subfigure about the spatial analysis in figure 3. It should tell people how much spatial information is encoded in each case. But I haven't found a proper subfigure design.
- [x] *July 15*: Add main content of figure 4.
- [ ] Add main content of figure 5 (*due July 16*)

**July 16**
- [ ] Check how the experimental data report the time cells change over time
- [ ] https://www.bu.edu/psych/profile/marc-howard-ph-d/, theoretical papers about time cells.
- [ ] Find better parameters for the first (time cell) experiment.
  - [ ] Adjust the interval between two temporal events to see if the time cells can be widened further.
  - [ ] Try to increase the trial duration from 5s to 10s? Use `torch.grad_clip`
  - [ ] Worst case, try GRU/LSTM first. Worst-case use State Space Model.
- [ ] For parameter search (https://hydra.cc/docs/intro/)
- [ ] Get some ideas on how to frame a qualitative explanation of the time cells.
