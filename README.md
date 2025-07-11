# STCell
The investigation of time cell and place cell unification in the hippocampus CA3 subfield.

## TODOs
- [done] Push code to GitHub.
- [done] Organize the code, try to keep the model training and the analyzing code separate.
- [ ] **Very Important** Do a sanity test on Zhaoze's experiment.
- [ ] Add more references to the intro.
- [ ] For figure 2 panel a,b,c. Add another (few) plots that indicate the reuse of time cells (hidden units) for the space task (similar to the current figure 2.b).
- [ ] For the ring plots, plot each ratemap's min, max, mean. (use np.nanmean/nanmin/nanmax)

### 2025-July-11

#### Time experiment: 

1. I have added the preactivation noise=0.4 and the post-activation noise=0.3 and a significant neurons fire sequentially during the delayed interval. 

2. Consider the loss = A mse loss +  B fr constraint.

- If \textbf{A=1, B=0.0001}, during training (loss converge + loss < 1), initial (A mse, B fr) = (0.18, 0.03), end of training (A mse, B fr) = (0.02, 0.003). Sequential firing is observed. 
![sequential firing of 1:0.0001](image-1.png)

- If A=1, B=1, during training (loss converge + loss < 1),, initial (A mse, B fr) = (0.19, 288), end of training (A mse, B fr) = (0.10, 0.08). It doesn't learn with 5000 epochs. 
![doesn't learn](image-2.png)

3. Fig 1 is completed. 