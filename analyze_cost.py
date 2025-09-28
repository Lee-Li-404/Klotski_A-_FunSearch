def analyze_costs(filename="klotski_levels_30_20250928_033009.txt"):
    buckets = {
        "30-40": [],
        "40-50": [],
        "50-60": [],
        "60-70": [],
        "70-80": [],
        "80+": []
    }

    current_name = None
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NAME "):
                current_name = line[5:].strip()
            elif line.startswith("BOARD "):
                # 没有 NAME 的情况
                if current_name is None:
                    current_name = "(no name)"
            elif line.startswith("COST "):
                try:
                    cost = int(line.split()[1])
                except:
                    cost = -1

                if cost < 30:
                    if cost == -1:
                        buckets["80+"].append(current_name)
                    else:
                        pass  # 跳过
                elif cost <= 40:
                    buckets["30-40"].append(current_name)
                elif cost <= 50:
                    buckets["40-50"].append(current_name)
                elif cost <= 60:
                    buckets["50-60"].append(current_name)
                elif cost <= 40:
                    buckets["60-70"].append(current_name)
                elif cost <= 80:
                    buckets["70-80"].append(current_name)
                else:
                    buckets["80+"].append(current_name)

                current_name = None  # reset，准备下一个局面

    print("📊 Cost 分布统计：")
    for k, v in buckets.items():
        print(f"{k}: {len(v)} 个")
        print("   IDs:", ", ".join(v) if v else "(none)")


if __name__ == "__main__":
    analyze_costs("klotski_levels_30_20250928_033009.txt")
