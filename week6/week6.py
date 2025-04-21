{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a357ef1c-f482-4723-8358-929cb6b0e120",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "558b20e1-4324-4a2d-89d6-d0b88ddd7261",
   "metadata": {},
   "outputs": [],
   "source": [
    "class GridWorld:\n",
    "    \n",
    "    def __init__(self, size):\n",
    "        self.size = size\n",
    "        self.actions = [\"up\", \"down\", \"left\", \"right\"]\n",
    "        \n",
    "        self.agent_pos = (random.randint(0,self.size -1), random.randint(0,self.size -1))\n",
    "        self.pickup_pos = (random.randint(0,self.size -1), random.randint(0,self.size -1))\n",
    "        self.dropoff_pos = (self.size-1, self.size-1)\n",
    "        self.loaded = False\n",
    "        self.done = False\n",
    "        \n",
    "\n",
    "    def _get_state(self):\n",
    "        return (self.agent_pos, self.loaded, self.pickup_pos) \n",
    "\n",
    "    def step(self, action):\n",
    "        if self.done:\n",
    "            return self._get_state(), 0, True\n",
    "\n",
    "        x,y = self.agent_pos \n",
    "\n",
    "        if action == \"up\" and x > 0:\n",
    "            x-=1\n",
    "        elif action == \"down\" and x < self.size-1:\n",
    "            x+=1\n",
    "        elif action == \"left\" and y > 0:\n",
    "            y-=1\n",
    "        elif action == \"right\" and y < self.size-1:\n",
    "            y+=1\n",
    "\n",
    "        self.agent_pos = (x,y)\n",
    "        reward = -1\n",
    "\n",
    "        if not self.loaded and self.agent_pos == self.pickup_pos:\n",
    "            self.loaded = True \n",
    "        elif self.agent_pos == self.dropoff_pos:\n",
    "            reward = 20\n",
    "            self.done = True \n",
    "\n",
    "        return self._get_state(), reward, self.done\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27e7e539-2132-4b78-acfc-3141cb3aaef8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
