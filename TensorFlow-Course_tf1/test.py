from typing import List


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not len(nums): return 0
        f = [1] * (len(nums) + 1)
        for i in range(2, len(nums)+1):
            max_num = float('-inf')
            nums_i = i-1
            for k in range(0, nums_i+1):
                if nums[nums_i] > nums[k]:
                    max_num = max(max_num, 1+f[k+1])
            if max_num != float('-inf'): f[i] = max_num 
        return f

sol = Solution()
nums = [10,9,2,5,3,7,101,18]
res = sol.lengthOfLIS(nums=nums)
print(res)