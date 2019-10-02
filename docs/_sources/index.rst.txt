======================================
qhoptim: Quasi-hyperbolic optimization
======================================

.. raw:: html

    <!-- Place this tag where you want the button to render. -->
    <a class="github-button" href="https://github.com/facebookresearch/qhoptim" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star facebookresearch/qhoptim on GitHub">Star</a>

    <!-- Place this tag where you want the button to render. -->
    <a class="github-button" href="https://github.com/facebookresearch/qhoptim/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork facebookresearch/qhoptim on GitHub">Fork</a>

    <!-- Place this tag where you want the button to render. -->
    <a class="github-button" href="https://github.com/facebookresearch/qhoptim/issues" data-icon="octicon-issue-opened" data-size="large" data-show-count="true" aria-label="Issue facebookresearch/qhoptim on GitHub">Issue</a>

----

The qhoptim library provides PyTorch and TensorFlow implementations of the
quasi-hyperbolic momentum (QHM) and quasi-hyperbolic Adam (QHAdam)
optimization algorithms from Facebook AI Research.

Quickstart
==========

Use this one-liner for installation::

    $ pip install qhoptim

Then, you can instantiate the optimizers in PyTorch:

.. doctest::

    >>> from qhoptim.pyt import QHM, QHAdam

    # something like this for QHM
    >>> optimizer = QHM(model.parameters(), lr=1.0, nu=0.7, momentum=0.999)

    # or something like this for QHAdam
    >>> optimizer = QHAdam(
    ...     model.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.995, 0.999))

Or in TensorFlow:

.. doctest::

    >>> from qhoptim.tf import QHMOptimizer, QHAdamOptimizer

    # something like this for QHM
    >>> optimizer = QHMOptimizer(
    ...     learning_rate=1.0, nu=0.7, momentum=0.999)

    # or something like this for QHAdam
    >>> optimizer = QHAdamOptimizer(
    ...     learning_rate=1e-3, nu1=0.7, nu2=1.0, beta1=0.995, beta2=0.999)

Please refer to the links on the menubar for detailed installation instructions
and API references.

Choosing QHM parameters
-----------------------

For those who use momentum or Nesterov's accelerated gradient with momentum
constant :math:`\beta = 0.9`, we recommend trying out QHM with
:math:`\nu = 0.7` and momentum constant :math:`\beta = 0.999`. You'll need to
normalize the learning rate by dividing by :math:`1 - \beta_{old}`.

Similarly, for those who use Adam with :math:`\beta_1 = 0.9`, we recommend
trying out QHAdam with :math:`\nu_1 = 0.7`, :math:`\beta_1 = 0.995`,
:math:`\nu_2 = 1`, and all other parameters unchanged.

Below is a handy widget to help convert from SGD with (Nesterov) momentum to
QHM:

.. raw:: html

   <style>
     .sgdnumhp {
       width: 75px !important;
     }

     .qhmnumhp {
       width: 75px !important;
     }

     .qhmnumhpcur {
       background-color: rgb(255, 220, 230) !important;
     }

     .qhmnumhpadv {
       background-color: rgb(199, 255, 199) !important;
     }

     .qhmadvisor {
       background-blend-mode: darken;
       background-color: rgba(0, 0, 0, .05);
       border-color: #2980B9;
       border-radius: 10px;
       border-style: solid;
       max-width: 470px;
       padding: 12px;
       margin-bottom: 25px;
     }

     .currentqhm {
       margin-top: 20px;
     }

     .advisedqhm {
       margin-top: 20px;
     }
   </style>

   <div class="qhmadvisor">
     <h5>
       QHM Hyperparameter Advisor
     </h5>
     <form>
       <b>Your current SGD learning rate (unnormalized):</b><br>
       <input class="sgdnumhp" type="text" id="field-sgdlr" value="0.1" oninput="refresh_qhm_advisor()"><br>
       <b>Your current SGD momentum:</b><br>
       <input class="sgdnumhp" type="text" id="field-sgdmom" value="0.9" oninput="refresh_qhm_advisor()"><br>
       <b>Are you using Nesterov?</b>
       <input type="checkbox" id="field-sgdnag"
        onchange="refresh_qhm_advisor()" checked><br>
     </form>

     <div class="currentqhm">
       <span>Your current SGD optimizer is equivalent to QHM with:</span><br>
       <b>QHM learning rate (alpha):</b><br>
       <input class="qhmnumhp qhmnumhpcur" type="text" id="currentqhmalpha" readonly><br>
       <b>QHM immediate discount (nu):</b><br>
       <input class="qhmnumhp qhmnumhpcur" type="text" id="currentqhmnu" readonly><br>
       <b>QHM momentum (beta):</b><br>
       <input class="qhmnumhp qhmnumhpcur" type="text" id="currentqhmbeta" readonly><br>
     </div>

     <div class="advisedqhm">
       <span>We suggest that you try the following QHM hyperparameters:</span><br>
       <b>QHM learning rate (alpha):</b><br>
       <input class="qhmnumhp qhmnumhpadv" type="text" id="advisedqhmalpha" readonly><br>
       <b>QHM immediate discount (nu):</b><br>
       <input class="qhmnumhp qhmnumhpadv" type="text" id="advisedqhmnu" readonly><br>
       <b>QHM momentum (beta):</b><br>
       <input class="qhmnumhp qhmnumhpadv" type="text" id="advisedqhmbeta" readonly><br>
     </div>
   </div>

   <script>
     function refresh_qhm_advisor() {
       var format_float = function(f) {
         return f.toFixed(9).replace(/\.?0*$/,"");
       };

       var set_values = function(currentqhmalpha, currentqhmnu, currentqhmbeta,
                                 advisedqhmalpha, advisedqhmnu, advisedqhmbeta) {
         document.getElementById("currentqhmalpha").value = currentqhmalpha;
         document.getElementById("currentqhmnu").value = currentqhmnu;
         document.getElementById("currentqhmbeta").value = currentqhmbeta;
         document.getElementById("advisedqhmalpha").value = advisedqhmalpha;
         document.getElementById("advisedqhmnu").value = advisedqhmnu;
         document.getElementById("advisedqhmbeta").value = advisedqhmbeta;
       };

       var inner = function() {
         var sgdlr = parseFloat(document.getElementById("field-sgdlr").value);
         var sgdmom = parseFloat(document.getElementById("field-sgdmom").value);
         var sgdnag = document.getElementById("field-sgdnag").checked;

         var currentalpha = sgdlr / (1.0 - sgdmom);
         var currentnu = sgdnag ? sgdmom : 1.0;
         var currentmom = sgdmom;

         var advisedalpha = currentalpha;
         var advisednu = Math.max(0.0, 3.0 * sgdmom - 2.0);
         var advisedmom = Math.max(0.999, sgdmom);

         set_values(
             format_float(currentalpha),
             format_float(currentnu),
             format_float(currentmom),
             format_float(advisedalpha),
             format_float(advisednu),
             format_float(advisedmom));
       };

       var set_error = function() {
         set_values("ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR");
       };

       try {
         inner();
       } catch(err) {
         console.log("QHM advisor error", err);
         set_error();
         return false;
       }

       return true;
     }
     refresh_qhm_advisor();
   </script>

Reference
=========

QHM and QHAdam were proposed in the ICLR 2019 paper
`"Quasi-hyperbolic momentum and Adam for deep learning"`__. We recommend
reading the paper for both theoretical insights into and empirical analyses of
the algorithms.

__ https://arxiv.org/abs/1810.06801

If you find the algorithms useful in your research, we ask that you cite the
paper as follows:

.. code-block:: bibtex

    @inproceedings{ma2019qh,
      title={Quasi-hyperbolic momentum and Adam for deep learning},
      author={Jerry Ma and Denis Yarats},
      booktitle={International Conference on Learning Representations},
      year={2019}
    }

GitHub
======

The project's GitHub repository can be found `here`__. Bugfixes and
contributions are very much appreciated!

__ https://github.com/facebookresearch/qhoptim

.. toctree::
    :maxdepth: 2
    :caption: Table of contents

    install
    pyt
    tf

Index
=====

:ref:`genindex`
