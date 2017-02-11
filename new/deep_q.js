// An agent is in state0 and does action0
// environment then assigns reward0 and provides new state, state1
// Experience nodes store all ref information, which is used in the
// Q-learning update step
function experience (state0, action0, reward0, state1) {
    var ref = {};
    ref.state0  = state0;
    ref.action0 = action0;
    ref.reward0 = reward0;
    ref.state1  = state1;
    return ref;
};


// A Brain object does all the magic.
// over time it receives some inputs and some rewards
// and its job is to set the outputs to maximize the expected reward
function brain (
    number_of_states, 
    number_of_actions, 
    temporal_window, 
    experience_size, 
    start_learn_threshold, 
    gamma,
    learning_steps_total, 
    learning_steps_burn_in, 
    epsilon_min, 
    epsilon_test_time, 
    random_action_distribution, 
    layer_definitions) {
    var ref = {};
    ref.number_of_states      =  number_of_states;
    ref.number_of_actions     = number_of_actions;
    // in number of time steps, of temporal memory
    // the ACTUAL input to the net will be (x,a) temporal_window times, and followed by current x
    // so to have no information from previous time step going into value function, set to 0.
    ref.temporal_window       =   temporal_window || 1;
    // size of experience replay memory
    ref.experience_size       =   experience_size || 30000;
    // number of examples in experience replay memory before we begin learning
    ref.start_learn_threshold = start_learn_threshold || Math.floor(Math.min(ref.experience_size * 0.1, 1000)); 
    // gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
    ref.gamma = gamma || 0.8;

    // number of steps we will learn for
    ref.learning_steps_total = learning_steps_total || 100000;
    // how many steps of the above to perform only random actions (in the beginning)?
    ref.learning_steps_burn_in = learning_steps_burn_in || 3000;
    // what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end
    ref.epsilon_min = epsilon_min || 0.05;
    // what epsilon to use at test time? (i.e. when learning is disabled)
    ref.epsilon_test_time = epsilon_test_time || 0.01;

    // advanced feature. Sometimes a random action should be biased towards some values
    // for example in flappy bird, we may want to choose to not flap more often
    if (random_action_distribution) {
        // ref better sum to 1 by the way, and be of length ref.number_of_actions
        ref.random_action_distribution = random_action_distribution;
        if (ref.random_action_distribution.length != number_of_actions) {
            throw new Error('random_action_distribution should be same length as number_of_actions.');
        }
        var sum = 0; 
        for (var i = 0; i < ref.random_action_distribution.length; i++) { 
            sum += ref.random_action_distribution[i]; 
        }
        if (Math.abs(sum - 1.0) > 0.0001) { 
            throw new Error('random_action_distribution should sum to 1!'); 
        }
    }

    // states that go into neural net to predict optimal action look as
    // x0,a0,x1,a1,x2,a2,...xt
    // ref variable controls the size of that temporal window. Actions are
    // encoded as 1-of-k hot vectors
    ref.net_inputs    = ref.number_of_states * ref.temporal_window + ref.number_of_actions * ref.temporal_window + ref.number_of_states;
    if (ref.temporal_window < 2) {
        throw new Error('temporal_window must be at least 2, even higher provides more context.')
    }
    ref.window_size   = Math.max(ref.temporal_window, 2); // must be at least 2, but if we want more context even more
    ref.state_window  = new Array(ref.window_size);
    ref.action_window = new Array(ref.window_size);
    ref.reward_window = new Array(ref.window_size);
    ref.net_window    = new Array(ref.window_size);

    ref.layer_definitions = layer_definitions;
    if (ref.layer_definitions.length < 2) { 
        throw new Error('layer_definitions must have at least 2 layers'); 
    }
    if (ref.layer_definitions[0].type != 'input') { 
        throw new Error('layer_definitions first layer must be input layer!'); 
    }
    if (ref.layer_definitions[ref.layer_definitions.length - 1].type != 'regression') { 
        throw new Error('layer_definitions last layer must be input regression!'); 
    }
    if (ref.layer_definitions[0].out_depth * ref.layer_definitions[0].out_sx * ref.layer_definitions[0].out_sy != ref.net_inputs) {
        throw new Error('layer_definitions Number of inputs must be number_of_states * temporal_window + number_of_actions * temporal_window + number_of_states!');
    }
    if (ref.layer_definitions[ref.layer_definitions.length - 1].number_of_neurons != ref.number_of_actions) {
       throw new Error('layer definitions Number of regression neurons should be equal to number_of_actions!');
    }

    // TODO CROSS FILE
    ref.value_net = new convnetjs.Net();
    ref.value_net.make_layers(ref.layer_definitions);
    // and finally we need a Temporal Difference Learning trainer!
    var tdtrainer_options = {
        learning_rate:0.01, 
        momentum:0.0, 
        batch_size:64, 
        l2_decay:0.01
    };
    if(typeof opt.tdtrainer_options !== 'undefined') {
        tdtrainer_options = opt.tdtrainer_options; // allow user to overwrite ref
    }
    ref.tdtrainer = new convnetjs.SGDTrainer(ref.value_net, tdtrainer_options);

    // experience replay
    ref.experience = [];

    ref.age                   = 0; // incremented every backward()
    ref.forward_passes        = 0; // incremented every forward()
    ref.epsilon               = 1.0; // controls exploration exploitation tradeoff. Should be annealed over time
    ref.latest_reward         = 0;
    ref.last_input_array      = [];
    ref.learning              = true;


    ref.random_action = function () {
        // a bit of a helper function. It returns a random action
        // we are abstracting ref away because in future we may want to 
        // do more sophisticated things. For example some actions could be more
        // or less likely at "rest"/default state.
        if (ref.random_action_distribution.length == 0) {
            // TODO CROSS FILE
            return convnetjs.randi(0, ref.number_of_actions);
        } else {
            // okay, lets do some fancier sampling:
            var p = convnetjs.randf(0, 1.0);
            var cumprob = 0.0;
            for(var i = 0; i < ref.number_of_actions; i++) {
                cumprob += ref.random_action_distribution[i];
                if (p < cumprob) { 
                    return i; 
                }
            }
        }
    };

    ref.policy = function (s) {
        // compute the value of doing any action in this state
        // and return the argmax action and its value
        // TODO CROSS FILE
        var svol = new convnetjs.Vol(1, 1, ref.net_inputs);
        svol.w = s;
        var action_values = ref.value_net.forward(svol);
        var maxi = 0; 
        var maxval = action_values.w[0];
        for (var i = 1; i < ref.number_of_actions; i++) {
            if (action_values.w[i] > maxval) { 
                maxi = i; 
                maxval = action_values.w[i]; 
            }
        }
        return {
            action: maxi, 
            value: maxval
        };
    };

    ref.get_network_input = function (xt) {
        // return s = (x,a,x,a,x,a,xt) state vector. 
        // It's a concatenation of last window_size (x,a) pairs and current state x
        var w = [];
        w = w.concat(xt); // start with current state
        // and now go backwards and append states and actions from history temporal_window times
        for(var k = 0; k < ref.temporal_window; k++) {
            // state
            w = w.concat(ref.state_window[ref.window_size - 1 - k]);
            // action, encoded as 1-of-k indicator vector. We scale it up a bit because
            // we dont want weight regularization to undervalue this information, as it only exists once
            var action1ofk = new Array(ref.number_of_actions);
            for(var q = 0; q < ref.number_of_actions; q++) {
                action1ofk[q] = 0.0;
            }
            action1ofk[ref.action_window[ref.window_size - 1 - k]] = 1.0 * ref.number_of_states; // why is this a * 1.0?
            w = w.concat(action1ofk);
        }
        return w;
    };

    ref.forward = function (input_array) {
        // compute forward (behavior) pass given the input neuron signals from body
        ref.forward_passes += 1;
        ref.last_input_array = input_array; // back ref up

        // create network input
        var action;
        if (ref.forward_passes > ref.temporal_window) {
            // we have enough to actually do something reasonable
            var net_input = ref.get_network_input(input_array);
            if (ref.learning) {
                // compute epsilon for the epsilon-greedy policy
                ref.epsilon = Math.min(1.0, Math.max(ref.epsilon_min, 1.0 - (ref.age - ref.learning_steps_burn_in) / (ref.learning_steps_total - ref.learning_steps_burn_in))); 
            } else {
                ref.epsilon = ref.epsilon_test_time; // use test-time value
            }
            var rf = convnetjs.randf(0,1);
            if(rf < ref.epsilon) {
                // choose a random action with epsilon probability
                action = ref.random_action();
            } else {
                // otherwise use our policy to make decision
                var maxact = ref.policy(net_input);
                action = maxact.action;
            }
        } else {
            // pathological case that happens first few iterations 
            // before we accumulate window_size inputs
            var net_input = [];
            action = ref.random_action();
        }

        // remember the state and action we took for backward pass
        ref.net_window.shift();
        ref.net_window.push(net_input);
        ref.state_window.shift(); 
        ref.state_window.push(input_array);
        ref.action_window.shift(); 
        ref.action_window.push(action);

        return action;
    };




    return ref;
};

    
    
    
    
    
  Brain.prototype = {
    ,
    backward: function(reward) {
      ref.latest_reward = reward;
      ref.average_reward_window.add(reward);
      ref.reward_window.shift();
      ref.reward_window.push(reward);
      
      if(!ref.learning) { return; } 
      
      // various book-keeping
      ref.age += 1;
      
      // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
      // (given that an appropriate number of state measurements already exist, of course)
      if(ref.forward_passes > ref.temporal_window + 1) {
        var e = new Experience();
        var n = ref.window_size;
        e.state0 = ref.net_window[n-2];
        e.action0 = ref.action_window[n-2];
        e.reward0 = ref.reward_window[n-2];
        e.state1 = ref.net_window[n-1];
        if(ref.experience.length < ref.experience_size) {
          ref.experience.push(e);
        } else {
          // replace. finite memory!
          var ri = convnetjs.randi(0, ref.experience_size);
          ref.experience[ri] = e;
        }
      }
      
      // learn based on experience, once we have some samples to go on
      // ref is where the magic happens...
      if(ref.experience.length > ref.start_learn_threshold) {
        var avcost = 0.0;
        for(var k=0;k < ref.tdtrainer.batch_size;k++) {
          var re = convnetjs.randi(0, ref.experience.length);
          var e = ref.experience[re];
          var x = new convnetjs.Vol(1, 1, ref.net_inputs);
          x.w = e.state0;
          var maxact = ref.policy(e.state1);
          var r = e.reward0 + ref.gamma * maxact.value;
          var ystruct = {dim: e.action0, val: r};
          var loss = ref.tdtrainer.train(x, ystruct);
          avcost += loss.loss;
        }
        avcost = avcost/ref.tdtrainer.batch_size;
        ref.average_loss_window.add(avcost);
      }
    },
    visSelf: function(elt) {
      elt.innerHTML = ''; // erase elt first
      
      // elt is a DOM element that ref function fills with brain-related information
      var brainvis = document.createElement('div');
      
      // basic information
      var desc = document.createElement('div');
      var t = '';
      t += 'experience replay size: ' + ref.experience.length + '<br>';
      t += 'exploration epsilon: ' + ref.epsilon + '<br>';
      t += 'age: ' + ref.age + '<br>';
      t += 'average Q-learning loss: ' + ref.average_loss_window.get_average() + '<br />';
      t += 'smooth-ish reward: ' + ref.average_reward_window.get_average() + '<br />';
      desc.innerHTML = t;
      brainvis.appendChild(desc);
      
      elt.appendChild(brainvis);
    }
  }
  
  global.Brain = Brain;
})(deepqlearn);
