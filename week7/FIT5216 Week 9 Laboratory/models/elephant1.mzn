int: T; % maximum time allowed;
set of int: TIME = 1..T;
int: G; % number of guards
int: F; 
int: W;

array[TIME] of var -F..F: move_food;      % move food +ve bank to boat, -ve boat to bank
array[TIME] of var -W..W: move_water;     % move water +ve bank to boat, -ve boat to bank
var TIME: end;      % end time;     


% array[0..T] of var 0..F : food_boat;
% array[0..T] of var 0..F : food_bank;
% array[0..T] of var 0..W : water_boat;
% array[0..T] of var 0..W : water_bank;
% array[0..T] of var POS: elephant_pos;


set of int: TIME0 = 0..T;
set of int: FOOD = 0..F;
set of int: WATER = 0..W;

enum POS = {bank, boat};
array[POS] of POS: opposite = [boat, bank];

array[TIME0] of var POS: pos; % elephant at time t
array[TIME0] of var FOOD: food;
array[TIME0] of var WATER: water;


% 1. After end time, no food or water movement;
constraint forall(i in TIME)(i > end -> (move_food[i] = 0 /\ move_water[i] = 0));

% 2. Elephant in bank at first 
constraint pos[0] = bank /\ food[0] = F /\ water[0] = W;


% 3. No food or water left in the boat at the end and position of elephant on boat
constraint pos[end] = boat /\ food[end] = F /\ water[end] = W;


% 4. Food & water at bank after moving at each time t in T
constraint forall(t in TIME)(food[t] = food[t-1] - move_food[t]);
constraint forall(t in TIME)(water[t] = water[t-1] - move_water[t]);

% 5. All the movement in the same step have to be in the same direction 
constraint forall(i in TIME)(move_food[i] * move_water[i] >= 0);


% 6. Food and water taken equal to max guard & as movement in the same step is in the same direction 
constraint forall(i in TIME)(abs(move_water[i] + move_water[i]) <= G);

% 7. Position of elephant based on consitionals at each time                           
% constraint forall(i in 1..T-1)(food_boat[i] >= food_bank[i] -> elephant_pos[i+1] = opposite[elephant_pos[i]]);
% constraint forall(i in 1..T-1)(water_boat[i] >= water_boat[i] -> elephant_pos[i+1] = opposite[elephant_pos[i]]);
% constraint forall(t in TIME)
%             (if pos[t-1]=1 then elephant(bank, food[t-1], water[t-1], pos[t])
%             else elephant(boat, F-food[t-1], W-water[t-1], pos[t]) endif);
            
% Predicates
% 1. Elephant movement conditions
predicate elephant_moves(var FOOD: food_here, var WATER: water_here) = 
                           let {var FOOD: food_there = F- food_here;
                                var WATER: water_there = W - water_here; } in 
                                 (food_there = water_there) /\ (food_there > food_here) /\ (water_there > water_here);
                                 

% 2. Position of the elephant conditions with eat if there is only food / water left where it is and it doesn't move
predicate elephant(var POS:pos, var FOOD: food_here, var WATER: water_here, var POS:new_pos) = 
                   let {var bool:b = elephant_moves(food_here, water_here);} in 
                   new_pos = [pos, opposite[pos]][b+1] /\ (b \/ (food_here = 0 <-> water_here = 0));


% same as the above line 
% predicate elephant(var POS:pos, var FOOD: food_here, var WATER: water_here, var POS:new_pos) = 
%                     if elephant_moves(food_here, water_here) then new_pos = opposite[pos] else
%                     new_pos = pos /\ (food_here = 0 <-> water_here = 0) endif;

constraint forall(t in TIME)
          (if pos[t-1] = 1 then elephant(bank, food[t-1],water[t-1], pos[t])
          else elephant(boat, F-food[t-1], W-water[t-1],pos[t]) endif);


% objective
solve minimize end;
