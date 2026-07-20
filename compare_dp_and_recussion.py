
# compare dynamic programming
def fibo(n):
    if n <= 1:
        return n;
    return fibo(n-1) + fibo(n-2)

print(fibo(10))

def fibo_dp(n):
    if n < 0:
        return "vui lòng nhập vào số >= 0"
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    dp = [0] * (n+1)
    print(dp)
    dp[0] = 0
    dp[1] = 1

    for i in range(2,n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

print(fibo_dp(10))
    
