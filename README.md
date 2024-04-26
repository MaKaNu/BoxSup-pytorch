# BoxSup-pytorch


[![PyPI version](https://badge.fury.io/py/boxsup-pytorch.svg)](https://badge.fury.io/py/boxsup-pytorch)
![versions](https://img.shields.io/pypi/pyversions/boxsup-pytorch.svg)
[![GitHub license](https://img.shields.io/github/license/makanu/boxsup-pytorch.svg)](https://github.com/makanu/boxsup-pytorch/blob/main/LICENSE)


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


The BoxSup-pytorch is an implementation of the BoxSup Algorithm in pytorch. Since the original paper has no available implementation this implementation is only based on the paper. They may be a few differences to the explanations from the paper.


- Free software: GPLv3
- Documentation: https://makanu.github.io/boxsup-pytorch.


## Features

* TODO

## Credits


This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`mgancita/cookiecutter-pypackage`](https://mgancita.github.io/cookiecutter-pypackage/) project template.


### AI Help

#### How to Vectorize pytorch

ME:
Ich habe folgende Tensoren: est_mask, lab_mask

est_masks hat folgende Form: NxMxWxH, mit N=Batchsize, M=numclasses, W=ImageWidth and H=ImageHeight
lab_mask hat folged Form: NxPxWxH, mit P=CandidateCount

folgendes Funktioniert um den CrossEntropy Loss zu bestimmen:

```
loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
temp = []
for i in range(100):
    temp.append(loss(est_mask, lab_mask[:,i,:,:].long()))
torch.stack(temp,dim=1) 
```
Anstatt dieses Vorgangs würde ich gerne vektorisieren und zwar so dass ähnlich wie im For-Loop jeder Kandidat (P) aus lab_mask mit est_mask berechnet wird.

AI:
Um den Vorgang zu vektorisieren und jeden Kandidaten aus lab_mask mit est_mask zu berechnen, kannst du das unsqueeze- und expand-Verfahren verwenden. Hier ist eine Möglichkeit, den Vorgang zu vektorisieren:

loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
est_mask_expanded = est_mask.unsqueeze(1).expand(-1, lab_mask.size(1), -1, -1, -1)
loss_values = loss(est_mask_expanded, lab_mask.long())

Hier ist eine Erläuterung dazu:

    unsqueeze(1) fügt eine neue Dimension hinzu, um est_mask mit der Anzahl der Kandidaten in lab_mask kompatibel zu machen.
    expand(-1, lab_mask.size(1), -1, -1, -1) erweitert est_mask auf die gleiche Größe wie lab_mask entlang der hinzugefügten Dimension.
    loss(est_mask_expanded, lab_mask.long()) berechnet den Cross-Entropy Loss zwischen est_mask_expanded und lab_mask. Da lab_mask bereits die richtige Form hat, ist keine weitere Indizierung oder Schleife erforderlich.

Die Variable loss_values wird dann die Loss-Werte für jeden Kandidaten in lab_mask enthalten.